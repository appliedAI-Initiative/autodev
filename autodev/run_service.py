from typing import Literal

import torch

from autodev import logging
from autodev.autocomplete.completion_model import CompletionModel
from autodev.autocomplete.model import SantaCoderModelFactory, ModelFactory, ModelTransformationBetterTransformer
from autodev.llm import LLMType
from autodev.service import Service


log = logging.getLogger(__name__)


def run_service(completion_model_name: Literal["santacoder-ruby", "santacoder", "starcoder"]):
    # create completion model
    completion_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if completion_model_name == "starcoder":
        completion_model_path = "bigcode/starcoder"
        completion_model_factory = ModelFactory(completion_model_path)
    else:
        completion_model_factory = SantaCoderModelFactory()
        if completion_model_name == "santacoder-ruby":
            completion_model_path = "models/checkpoints/ruby/checkpoint-6000"
        elif completion_model_name == "santacoder":
            completion_model_path = "bigcode/santacoder"
        else:
            raise ValueError(completion_model_name)
    max_completion_tokens = 32
    log.info(f"Loading completion model '{completion_model_path}'")
    completion_model = CompletionModel(completion_model_factory.create_model(completion_model_path),
        completion_model_factory.create_tokenizer(), device=completion_device, max_new_tokens=max_completion_tokens)

    Service(LLMType.OPENAI_CHAT_GPT4, completion_model).run()


if __name__ == '__main__':
    logging.configure()

    run_service("santacoder-ruby")
    #run_service("starcoder")
