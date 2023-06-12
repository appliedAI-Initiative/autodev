"""
Fine-Tune SantaCoder on code/text dataset
"""

import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

import jsonargparse
import numpy as np
import peft
import torch
from datasets import load_dataset, Dataset
from peft import TaskType
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging as tflogging,
    set_seed, PrinterCallback, TrainerState, WEIGHTS_NAME,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TRAINING_ARGS_NAME

import fim

log = logging.getLogger(__name__)


@dataclass
class FineTuningConfiguration:
    model_path: str = "bigcode/santacoder"
    dataset_name: str = "bigcode/the-stack-dedup"
    subset: str = "data"
    split: str = "train"
    size_valid_set: int = 4000
    streaming: bool = False
    shuffle_buffer: int = 5000
    data_column: str = "content"
    seq_length: int = 1024
    max_steps: int = 10000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    eos_token_id: int = 49152
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    weight_decay: float = 0.05
    local_rank: int = 0
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 0
    num_workers: int = None
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: bool = False
    log_freq: int = 1
    eval_freq: int = 1000
    save_freq: int = 1000
    fim_rate: float = 0
    fim_spm_rate: float = 0
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 8
    lora_target_modules: Optional[List[str]] = None
    lora_dropout = 0.1


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            cfg (FineTuningConfiguration): the configuration
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """
    def __init__(
        self,
        cfg: FineTuningConfiguration,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id else cfg.eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = fim.get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            # obtain buffer where each element contains code as text
            # buffer_len is the total length (in characters) of all buffer elements combined
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            # tokenize the buffer elements
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]

            # for each buffer element, either
            #    * use the tokens directly or
            #    * (with some probability) use a transformed version for the fill in the middle (FIM) task
            #      (by selecting two random split points to split the sequence into prefix, middle and suffix,
            #      adding separator tokens in between)
            all_token_ids = []
            np_rng = np.random.RandomState(seed=self.seed)
            for tokenized_input in tokenized_inputs:

                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = fim.permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            # extract (non-overlapping) subsequences of length seq_length from all_token_ids
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                        "input_ids": torch.LongTensor(example),
                        "labels": torch.LongTensor(example),
                    }


def load_train_val_datasets(dataset_name, data_dir, split="train", size_valid_set=4000, streaming=False, seed=0,
        shuffle_buffer=5000, num_workers=None) -> Tuple[Dataset, Dataset]:
    dataset = load_dataset(
        dataset_name,
        data_dir=data_dir,
        split=split,
        use_auth_token=True,
        num_proc=num_workers if not streaming else None,
        streaming=streaming,
    )
    if streaming:
        log.info("Loading the dataset in streaming mode")
        valid_data = dataset.take(size_valid_set)
        train_data = dataset.skip(size_valid_set)
        train_data = train_data.shuffle(buffer_size=shuffle_buffer, seed=seed)
    else:
        # TODO: This should actually use size_valid_set, but all models were trained with this parametrisation
        # and we need to retain this (for now) in order for the evaluation, which uses the same split,
        # to be sound
        dataset = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
    return train_data, valid_data


class LoggingCallback(PrinterCallback):
    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            log.info(f"Step {state.global_step}: {logs}")


class LoraCompatibleTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # NOTE: The code below is a copy of the original super-class implementation where the instance check
        # for `PreTrainedModel` is replaced by a duck-typing-style check which simply checks for the
        # presence of method `save_pretrained`, which will work for regular models as well as PEFT models.

        def is_pretrained_model(m):
            # return isinstance(m, PreTrainedModel)  # <- original implementation
            return hasattr(m, "save_pretrained")

        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not is_pretrained_model(self.model):
            if is_pretrained_model(unwrap_model(self.model)):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                log.warning("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Addition for PeftModels in order to enable checkpoint resumption:
        # If the state dict/weights have not been saved, save them now
        weights_path = os.path.join(output_dir, WEIGHTS_NAME)
        if not os.path.exists(weights_path):
            torch.save(self.model.state_dict(), weights_path)


class CompletionFineTuning:
    def __init__(self, cfg: FineTuningConfiguration):
        self.cfg = cfg

    def create_datasets(self, tokenizer):
        cfg = self.cfg
        train_data, valid_data = load_train_val_datasets(cfg.dataset_name, cfg.subset, split=cfg.split,
            streaming=cfg.streaming, seed=cfg.seed, shuffle_buffer=cfg.shuffle_buffer, num_workers=cfg.num_workers,
            size_valid_set=cfg.size_valid_set)
        if not cfg.streaming:
            log.info(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
        chars_per_token = chars_token_ratio(train_data, tokenizer, cfg.data_column)
        log.info(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
        train_dataset = ConstantLengthDataset(
            cfg,
            tokenizer,
            train_data,
            infinite=True,
            seq_length=cfg.seq_length,
            chars_per_token=chars_per_token,
            content_field=cfg.data_column,
            fim_rate=cfg.fim_rate,
            fim_spm_rate=cfg.fim_spm_rate,
            seed=cfg.seed,
        )
        valid_dataset = ConstantLengthDataset(
            cfg,
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=cfg.seq_length,
            chars_per_token=chars_per_token,
            content_field=cfg.data_column,
            fim_rate=cfg.fim_rate,
            fim_spm_rate=cfg.fim_spm_rate,
            seed=cfg.seed,
        )

        return train_dataset, valid_dataset

    def run(self):
        cfg = self.cfg
        log.info(f"Running with {cfg}")

        set_seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

        tflogging.set_verbosity_info()

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_auth_token=True)

        train_data, val_data = self.create_datasets(tokenizer)

        log.info("Loading the model")
        # disable caching mechanism when using gradient checkpointing
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            trust_remote_code=True,
            use_cache=not cfg.gradient_checkpointing,
        )
        train_data.start_iteration = 0

        run_name = f"santacoder-{cfg.subset}"

        if cfg.use_lora:
            run_name += "-lora"
            peft_cfg = peft.LoraConfig(
                target_modules=cfg.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
            )
            model = peft.get_peft_model(model, peft_cfg)
            model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            dataloader_drop_last=True,
            evaluation_strategy="steps",
            max_steps=cfg.max_steps,
            eval_steps=cfg.eval_freq,
            save_steps=cfg.save_freq,
            save_total_limit=None,
            logging_steps=cfg.log_freq,
            log_level="debug",
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_steps=cfg.num_warmup_steps,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            weight_decay=cfg.weight_decay,
            run_name=run_name,
            report_to=["mlflow"],
            disable_tqdm=True,
            logging_nan_inf_filter=False,
        )

        trainer = LoraCompatibleTrainer(
            model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data,
            callbacks=[LoggingCallback()]
        )

        log.info("Training...")
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

        log.info("Saving last checkpoint of the model")
        model.save_pretrained(os.path.join(cfg.output_dir, "final_checkpoint/"))


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    cfg = jsonargparse.CLI(FineTuningConfiguration, as_positional=False)
    CompletionFineTuning(cfg).run()
