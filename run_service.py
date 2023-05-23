from autodev.llm import LLMType
from autodev.service import Service


if __name__ == '__main__':
    Service(LLMType.OPENAI_CHAT_GPT4).run()