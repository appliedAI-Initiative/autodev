from autodev import logging
from autodev.autocomplete.model import SantaCoderModelFactory
from autodev.llm import LLMType
from autodev.service import Service


if __name__ == '__main__':
    logging.configure()
    Service(LLMType.OPENAI_CHAT_GPT4, SantaCoderModelFactory(), "bigcode/santacoder", "cuda:0").run()