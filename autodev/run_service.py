import os
import torch

from autodev import logging
from autodev.autocomplete.model import SantaCoderModelFactory
from autodev.llm import LLMType
from autodev.service import Service


if __name__ == '__main__':
    logging.configure()

    checkpoint = "../remote-santacoder-finetuning/checkpoints/ruby/checkpoint-6000"
    if os.path.exists(checkpoint):
        completion_model = checkpoint
    else:
        completion_model = "bigcode/santacoder"
    completion_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_completion_tokens = 32

    Service(LLMType.OPENAI_CHAT_GPT4, SantaCoderModelFactory(), completion_model, completion_device,
        max_completion_tokens=max_completion_tokens).run()