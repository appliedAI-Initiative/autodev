from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from langchain import OpenAI, HuggingFacePipeline
from langchain.llms import OpenAIChat, BaseLLM
from langchain.schema import LLMResult


class LLMFactory(ABC):
    @abstractmethod
    def create_llm(self) -> BaseLLM:
        pass


class LLMFactoryOpenAICompletions(LLMFactory):
    def __init__(self, model_name: Literal["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"] = "text-davinci-003"):
        self.model_name = model_name

    def create_llm(self) -> OpenAI:
        return OpenAI(model_name=self.model_name)


class LLMFactoryOpenAIChat(LLMFactory):
    def __init__(self, model_name: Literal["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"] = "gpt-4"):
        self.model_name = model_name

    def create_llm(self) -> OpenAIChat:
        return OpenAIChat(model_name=self.model_name)


class LLMFactoryHuggingFace(LLMFactory):
    def __init__(self, model: str, task=None, trust_remote_code=True, device_map="auto", return_full_text=True,
            tokenizer=None, **kwargs):
        self.kwargs = kwargs
        self.kwargs.update(dict(task=task, model=model, trust_remote_code=trust_remote_code, device_map=device_map,
            tokenizer=tokenizer, return_full_text=return_full_text))

    def create_llm(self) -> HuggingFacePipeline:
        from transformers import pipeline
        pipe = pipeline(**self.kwargs)
        return HuggingFacePipeline(pipeline=pipe)


class LLMFactoryHuggingFaceGPT2(LLMFactoryHuggingFace):
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        super().__init__(task="text-generation", model=model, tokenizer=tokenizer)


class LLMFactoryHuggingFaceDolly7B(LLMFactoryHuggingFace):
    def __init__(self):
        super().__init__("databricks/dolly-v2-7b")


class LLMType(Enum):
    OPENAI_DAVINCI3 = LLMFactoryOpenAICompletions
    OPENAI_CHAT_GPT4 = LLMFactoryOpenAIChat
    HUGGINGFACE_GPT2 = LLMFactoryHuggingFaceGPT2
    HUGGINGFACE_DOLLY_7B = LLMFactoryHuggingFaceDolly7B

    def create_factory(self) -> LLMFactory:
        factory_cls = self.value
        return factory_cls()

    def create_llm(self) -> BaseLLM:
        return self.create_factory().create_llm()

    def chunk_size(self):
        if self in (self.OPENAI_DAVINCI3, self.OPENAI_CHAT_GPT4):
            return 1000
        elif self in (self.HUGGINGFACE_GPT2, self.HUGGINGFACE_DOLLY_7B):
            return 512
        else:
            raise ValueError(self)


class TextInTextOut:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def query(self, prompt: str) -> str:
        result: LLMResult = self.llm.generate([prompt])
        return result.generations[0][0].text
