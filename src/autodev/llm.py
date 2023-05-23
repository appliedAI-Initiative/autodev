import queue
import threading
from abc import ABC, abstractmethod
from enum import Enum
from threading import Thread
from typing import Literal, Dict, Any, List, Union, Callable, Iterator, Optional

from langchain import OpenAI, HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import OpenAIChat, BaseLLM
from langchain.schema import LLMResult, AgentFinish, AgentAction
from transformers import Pipeline, pipeline, TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM


class Prompt:
    def __init__(self, prompt_text):
        self.text = prompt_text


class PromptFactory(ABC):
    @abstractmethod
    def create(self, prompt: str) -> Prompt:
        pass


class PromptFactoryIdentity(PromptFactory):
    def create(self, prompt: str) -> Prompt:
        return Prompt(prompt)


class StreamingLLM(ABC):
    def __init__(self, llm: BaseLLM, prompt_factory: Optional[PromptFactory] = None):
        self.llm = llm
        self.prompt_factory = prompt_factory if prompt_factory is not None else PromptFactoryIdentity()

    def _prompt(self, prompt: Union[str, Prompt]) -> Prompt:
        if not isinstance(prompt, Prompt):
            return self.prompt_factory.create(prompt)
        else:
            return prompt

    def query(self, prompt: Union[str, Prompt]) -> str:
        result: LLMResult = self.llm.generate([self._prompt(prompt).text])
        return result.generations[0][0].text

    @abstractmethod
    def query_streaming(self, prompt: Union[str, Prompt]) -> Iterator[str]:
        pass


class LLMFactory(ABC):
    @abstractmethod
    def create_llm(self, **kwargs) -> BaseLLM:
        """
        :param kwargs: parameters to pass on to langchain class (subclass of BaseLLM)
        :return: an instance of BaseLLM
        """
        pass

    @abstractmethod
    def create_streaming_llm(self) -> StreamingLLM:
        pass

    def create_prompt_factory(self) -> Optional[PromptFactory]:
        return None


class LLMFactoryOpenAICompletions(LLMFactory):
    def __init__(self, model_name: Literal["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001",
            "text-ada-001"] = "text-davinci-003"):
        self.model_name = model_name

    def create_llm(self, **kwargs) -> OpenAI:
        return OpenAI(model_name=self.model_name, **kwargs)

    def create_streaming_llm(self) -> StreamingLLM:
        llm = self.create_llm(streaming=True)
        return LangChainStreamingLLM(llm, prompt_factory=self.create_prompt_factory())


class LLMFactoryOpenAIChat(LLMFactory):
    def __init__(self, model_name: Literal["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301"] = "gpt-4"):
        self.model_name = model_name

    def create_llm(self, **kwargs) -> OpenAIChat:
        return OpenAIChat(model_name=self.model_name, **kwargs)

    def create_streaming_llm(self) -> StreamingLLM:
        llm = self.create_llm(streaming=True)
        return LangChainStreamingLLM(llm, prompt_factory=self.create_prompt_factory())


class LLMFactoryHuggingFace(LLMFactory):
    def __init__(self, model: str, task="text-generation", trust_remote_code=True, device_map="auto",
            return_full_text=True,
            tokenizer=None, streamer=None, **pipeline_args):
        self.pipeline_args = pipeline_args
        self.pipeline_args.update(dict(task=task, model=model, trust_remote_code=trust_remote_code,
            device_map=device_map,
            tokenizer=tokenizer, return_full_text=return_full_text, streamer=streamer))

    def tokenizer(self):
        tokenizer = self.pipeline_args["tokenizer"]
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.pipeline_args["model"])
            self.pipeline_args["tokenizer"] = tokenizer
        return tokenizer

    def create_pipeline(self, **kwargs) -> Pipeline:
        self.pipeline_args.update(kwargs)
        return pipeline(**self.pipeline_args)

    def create_llm(self, pipeline=None, **kwargs) -> HuggingFacePipeline:
        if pipeline is None:
            pipeline = self.create_pipeline()
        return HuggingFacePipeline(pipeline=pipeline, **kwargs)

    def create_streaming_llm(self) -> StreamingLLM:
        return TransformersStreamingLLM(self, prompt_factory=self.create_prompt_factory())


class LLMFactoryHuggingFaceGPT2(LLMFactoryHuggingFace):
    def __init__(self, **kwargs):
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)


class LLMFactoryHuggingFaceDolly7B(LLMFactoryHuggingFace):
    def __init__(self):
        super().__init__("databricks/dolly-v2-7b")


class LLMFactoryHuggingFaceStarChat(LLMFactoryHuggingFace):
    def __init__(self):
        super().__init__("HuggingFaceH4/starchat-alpha", max_new_tokens=1024)

    def create_prompt_factory(self) -> Optional[PromptFactory]:
        return self.PromptFactory()

    class PromptFactory(PromptFactory):
        def create(self, prompt: str) -> Prompt:
            return Prompt(f"<|system|><|end|><|user|>{prompt}<|end|><|assistant|>")


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


class LangChainStreamingLLM(StreamingLLM):
    def __init__(self, llm: BaseLLM, prompt_factory: Optional[PromptFactory] = None):
        super().__init__(llm, prompt_factory=prompt_factory)

    def query_streaming(self, prompt: Union[str, Prompt]) -> Iterator[str]:
        yield from self.TextInIteratorOut(self.llm).query(self._prompt(prompt).text)

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


class TransformersStreamingLLM(StreamingLLM):
    def __init__(self, factory: LLMFactoryHuggingFace, prompt_factory: Optional[PromptFactory] = None):
        streamer = TextIteratorStreamer(factory.tokenizer(), skip_prompt=True)
        # TODO The use of a single streamer becomes problematic when handling parallel requests.
        # We should investigate ways of injecting a new streamer for every request.
        pipeline = factory.create_pipeline(streamer=streamer)
        llm = factory.create_llm(pipeline=pipeline)
        self.streamer = streamer
        super().__init__(llm, prompt_factory=prompt_factory)

    def query_streaming(self, prompt: Union[str, Prompt]) -> Iterator[str]:
        def query():
            self.query(prompt)

        thread = Thread(target=query)
        thread.start()

        for token in self.streamer:
            if token == self.streamer.stop_signal:
                break
            yield token


class LLMType(Enum):
    OPENAI_DAVINCI3 = LLMFactoryOpenAICompletions
    OPENAI_CHAT_GPT4 = LLMFactoryOpenAIChat
    HUGGINGFACE_GPT2 = LLMFactoryHuggingFaceGPT2
    HUGGINGFACE_DOLLY_7B = LLMFactoryHuggingFaceDolly7B
    HUGGINGFACE_STARCHAT = LLMFactoryHuggingFaceStarChat

    def create_factory(self) -> LLMFactory:
        factory_cls = self.value
        return factory_cls()

    def create_llm(self) -> BaseLLM:
        return self.create_factory().create_llm()

    def create_streaming_llm(self) -> StreamingLLM:
        return self.create_factory().create_streaming_llm()

    def chunk_size(self):
        if self in (self.OPENAI_DAVINCI3, self.OPENAI_CHAT_GPT4):
            return 1000
        elif self in (self.HUGGINGFACE_GPT2, self.HUGGINGFACE_DOLLY_7B):
            return 512
        else:
            raise ValueError(self)
