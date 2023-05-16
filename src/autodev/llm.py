import queue
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, Dict, Any, List, Union, Callable, Iterator

from langchain import OpenAI, HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import OpenAIChat, BaseLLM
from langchain.schema import LLMResult, AgentFinish, AgentAction


class LLMFactory(ABC):
    @abstractmethod
    def create_llm(self, **kwargs) -> BaseLLM:
        pass

    def create_streaming_llm(self) -> BaseLLM:
        return self.create_llm(streaming=True)


class LLMFactoryOpenAICompletions(LLMFactory):
    def __init__(self, model_name: Literal["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"] = "text-davinci-003"):
        self.model_name = model_name

    def create_llm(self, **kwargs) -> OpenAI:
        return OpenAI(model_name=self.model_name, **kwargs)


class LLMFactoryOpenAIChat(LLMFactory):
    def __init__(self, model_name: Literal["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"] = "gpt-4"):
        self.model_name = model_name

    def create_llm(self, **kwargs) -> OpenAIChat:
        return OpenAIChat(model_name=self.model_name, **kwargs)


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


class TokenReaderCallback(BaseCallbackHandler):
    def __init__(self, on_token: Callable[[str | None], None]):
        """
        :param on_token: function to call whenever a new token is received; None is passed if the end of the response is
            reached
        """
        self.on_token = on_token

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.on_token(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.on_token(None)

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        pass

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        pass

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        pass

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        pass


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

    def create_streaming_llm(self) -> BaseLLM:
        return self.create_factory().create_streaming_llm()

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


class TextInIteratorOut:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    class ReaderThread(threading.Thread):
        def __init__(self, llm: BaseLLM, prompt: str, q: queue.Queue):
            super().__init__()
            self.llm = llm
            self.prompt = prompt
            self.queue = q

        def run(self):
            self.llm(self.prompt, callbacks=[TokenReaderCallback(lambda token: self.queue.put(token))])

    def query(self, prompt: str) -> Iterator[str]:
        q = queue.Queue()
        thread = self.ReaderThread(self.llm, prompt, q)
        thread.start()

        while True:
            token = q.get()
            if token is None:
                break
            yield token
