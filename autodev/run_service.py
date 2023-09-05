from typing import Literal

from autodev.autocomplete.completion_model import CompletionModel
from autodev.autocomplete.model import SantaCoderModelFactory, BigCodeModelFactory
from autodev.llm import LLMType
from autodev.service import Service
from autodev.util import logging

log = logging.getLogger(__name__)


def run_service(completion_model_name: Literal["santacoder-ruby", "santacoder", "starcoder"], chat_model_llmtype=LLMType.OPENAI_CHAT_GPT4,
        max_completion_tokens=32):
    # create completion model
    if completion_model_name == "starcoder":
        completion_model_path = "bigcode/starcoder"
        completion_model_factory = BigCodeModelFactory(completion_model_path)
    else:
        completion_model_factory = SantaCoderModelFactory()
        if completion_model_name == "santacoder-ruby":
            completion_model_path = "models/checkpoints/ruby/checkpoint-6000"
        elif completion_model_name == "santacoder":
            completion_model_path = "bigcode/santacoder"
        else:
            raise ValueError(completion_model_name)
    log.info(f"Loading completion model '{completion_model_path}'")
    completion_model = CompletionModel.from_model_factory(completion_model_factory, model_path=completion_model_path,
        max_tokens=max_completion_tokens)

    # run service
    Service(chat_model_llmtype, completion_model).run()


if __name__ == '__main__':
    logging.configure()

    run_service("santacoder")
    #run_service("santacoder-ruby")
    #run_service("starcoder")
