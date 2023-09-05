"""
Performs fine-tuning of an auto-completion model, teaching the model a new language
"""

import multiprocessing
import os
from glob import glob
from typing import Optional

from autodev.autocomplete.fim_config import BigCodeFIMTokens
from autodev.util import logging
from autodev.autocomplete.finetuning import FineTuningConfiguration, CompletionFineTuning

log = logging.getLogger(__name__)


def run_finetuning_santacoder_thestack(lang_id,
        max_steps=30000,
        eval_freq=500,
        save_freq=500,
        log_freq=1,
        fim_rate=0.5,
        fim_spm_rate=0.0,
        resume_from_checkpoint: Optional[bool] = None,
        use_lora=False,
        lora_r=32,
        fp16=True,
        output_dir: Optional[str] = None):

    task_name = lang_id
    if use_lora:
        task_name += f"-lora{lora_r}"

    if output_dir is None:
        output_dir = f"models/checkpoints/{task_name}"

    # determine if we should resume from checkpoint (if unspecified)
    if resume_from_checkpoint is None:
        checkpoints_exist = len(glob(os.path.join(output_dir, "checkpoint-*"))) > 0
        if checkpoints_exist:
            log.info("Found at least one checkpoint, so resuming from checkpoint")
        else:
            log.info(f"No checkpoints found in {output_dir}, will train without resuming from checkpoint")
        resume_from_checkpoint = checkpoints_exist

    # Create configuration that works for bigcode/santacoder on the aai VMs (V100 w/ 32 GB VRAM)
    if use_lora:
        # When using LoRA, gradient checkpointing cannot be used because of this issue:
        # https://github.com/huggingface/transformers/issues/23170
        # Not using gradient checkpointing results in too much memory usage when using batch size 2,
        # so we reduce it to 1
        gradient_checkpointing = False
        batch_size = 1
    else:
        gradient_checkpointing = True
        batch_size = 2
    cfg = FineTuningConfiguration(
        model_path="bigcode/santacoder",
        dataset_name="bigcode/the-stack-dedup",
        subset=f"data/{lang_id}",
        data_column="content",
        split="train",
        seq_length=2048,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_warmup_steps=500,
        eval_freq=eval_freq,
        save_freq=save_freq,
        log_freq=log_freq,
        num_workers=multiprocessing.cpu_count(),
        fim_tokens=BigCodeFIMTokens(),
        fim_rate=fim_rate,
        fim_spm_rate=fim_spm_rate,
        output_dir=output_dir,
        resume_from_checkpoint= resume_from_checkpoint,
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_target_modules=["kv_attn", "q_attn"])

    CompletionFineTuning(cfg).run()


if __name__ == '__main__':
    logging.configure()

    run_finetuning_santacoder_thestack("ruby")
    #run_finetuning_santacoder_thestack("ruby", use_lora=True, lora_r=16, fp16=False, output_dir="models/checkpoints/ruby-lora16-fp32", save_freq=500, eval_freq=1000)
    #run_finetuning_santacoder_thestack("ruby", use_lora=True, lora_r=64, fp16=False, output_dir="models/checkpoints/ruby-lora64-fp32", save_freq=500, eval_freq=1000)
    #run_finetuning_santacoder_thestack("csharp")
    #run_finetuning_santacoder_thestack("rust")
