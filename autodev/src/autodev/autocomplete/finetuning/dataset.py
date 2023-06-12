import logging
import random
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm

import fim

log = logging.getLogger(__name__)


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
        concat_token_id: int,
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
        self.concat_token_id = concat_token_id
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
