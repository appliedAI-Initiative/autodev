from enum import Enum

from langchain import OpenAI, HuggingFacePipeline
from langchain.llms import OpenAIChat, BaseLLM
import torch
from langchain.schema import LLMResult


class LLMType(Enum):
    OPENAI_DAVINCI3 = "OpenAI"
    OPENAI_CHAT_GPT4 = "OpenAIChatGPT4"
    HUGGINGFACE_GPT2 = "HF_GPT2"
    HUGGINGFACE_DOLLY_7B = "HF_DOLLY_7B"
    HUGGINGFACE_DOLLY_12B = "HF_DOLLY_12B"
    HUGGINGFACE_DOLLY_3B = "HF_DOLLY_3B"

    def create_llm(self) -> BaseLLM:
        if self == self.OPENAI_DAVINCI3:
            return OpenAI()
        elif self == self.OPENAI_CHAT_GPT4:
            return OpenAIChat(model_name="gpt-4")
        elif self == self.HUGGINGFACE_GPT2:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
            return HuggingFacePipeline(pipeline=pipe)
        elif self in (self.HUGGINGFACE_DOLLY_7B, self.HUGGINGFACE_DOLLY_12B, self.HUGGINGFACE_DOLLY_3B):
            from transformers import pipeline
            if self == self.HUGGINGFACE_DOLLY_12B:
                model = "databricks/dolly-v2-12b"
            elif self == self.HUGGINGFACE_DOLLY_7B:
                model = "databricks/dolly-v2-7b"
            elif self == self.HUGGINGFACE_DOLLY_3B:
                model = "databricks/dolly-v2-3b"
            else:
                raise ValueError(self)
            pipe = pipeline(
                model=model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                return_full_text=True,
                batch_size=16)
            return HuggingFacePipeline(pipeline=pipe)
        else:
            raise ValueError(self)

    def chunk_size(self):
        if self == self.OPENAI_DAVINCI3:
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
