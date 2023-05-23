"""
Provides abstractions for large language models (LLMs).
The high-level abstraction, which provides streaming-based queries, is given by the `StreamingLLM` class.
Instances of `StreamingLLM` can be created via specializations of the `LLMFactory` class.
`LLMType` constitutes a convenient enumeration of the model types considered in concrete factory implementations.
"""
import logging
import queue
import threading
from abc import ABC, abstractmethod
from enum import Enum
from threading import Thread
from typing import Literal, Dict, Any, List, Union, Callable, Iterator, Optional

import torch
from langchain import OpenAI, HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import OpenAIChat, BaseLLM
from langchain.schema import LLMResult, AgentFinish, AgentAction
from transformers import Pipeline, pipeline, TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

log = logging.getLogger(__name__)


class Prompt:
    """
    Represents a prompt that is fully prepared and suitable as input for a particular model.

    It is a mere wrapper around a string, which serves only to establish a separate representation for
    prepared prompts (in contrast to raw strings, which represent the original, unprepared prompt).
    """
    def __init__(self, prompt_text):
        self.text = prompt_text


class PromptFactory(ABC):
    @abstractmethod
    def create(self, prompt: str) -> Prompt:
        """
        Creates a prepared prompt for the given raw prompt
        :param prompt: the raw prompt
        :return: the prepared prompt which is suitable as input to a particular model
        """
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

    def query(self, prompt: Union[str, Prompt], stream=None) -> str:
        """
        Query the model with the given prompt, returning the full response string

        :param prompt: the prompt
        :param stream: a stream supporting the `write` method to which response tokens shall be written as they
            are received; if not None, the model's `query_streaming` method is used to generate the response
            rather than a non-streaming method of the base model
        :return: the full response string
        """
        if stream is None:
            result: LLMResult = self.llm.generate([self._prompt(prompt).text])
            return result.generations[0][0].text
        else:
            response = ""
            for token in self.query_streaming(prompt):
                response += token
                stream.write(token)
            return response

    @abstractmethod
    def query_streaming(self, prompt: Union[str, Prompt]) -> Iterator[str]:
        """
        Query the model with the given prompt, returning an iterator of response sub-strings/tokens,
        which, when concatenated, constitute the full response string

        :param prompt: the prompt
        :return: an iterator of response tokens
        """
        pass


class LLMFactory(ABC):
    @abstractmethod
    def create_llm(self, **kwargs) -> BaseLLM:
        """
        Creates a langchain LLM model

        :param kwargs: parameters to pass on to the langchain class (subclass of BaseLLM)
        :return: an instance of BaseLLM
        """
        pass

    @abstractmethod
    def create_streaming_llm(self) -> StreamingLLM:
        """
        :return: an instance of StreamingLLM, which supports a high-level streaming queries
        """
        pass

    def create_prompt_factory(self) -> Optional[PromptFactory]:
        """
        :return: the prompt factory to use for model instances or None if no prompt adaptation is required
        """
        return None


class LLMFactoryOpenAICompletions(LLMFactory):
    def __init__(self, model_name: Literal["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001",
            "text-ada-001"] = "text-davinci-003"):
        self.model_name = model_name

    def create_llm(self, **kwargs) -> OpenAI:
        return OpenAI(model_name=self.model_name, **kwargs)

    def create_streaming_llm(self) -> StreamingLLM:
        llm = self.create_llm(streaming=True)
        return StreamingLLMLangChainCallback(llm, prompt_factory=self.create_prompt_factory())


class LLMFactoryOpenAIChat(LLMFactory):
    def __init__(self, model_name: Literal["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301"] = "gpt-4"):
        self.model_name = model_name

    def create_llm(self, **kwargs) -> OpenAIChat:
        return OpenAIChat(model_name=self.model_name, **kwargs)

    def create_streaming_llm(self) -> StreamingLLM:
        llm = self.create_llm(streaming=True)
        return StreamingLLMLangChainCallback(llm, prompt_factory=self.create_prompt_factory())


class LLMFactoryHuggingFace(LLMFactory):
    def __init__(self, model: str, task="text-generation", trust_remote_code=True, device_map="auto",
            return_full_text=True,
            tokenizer=None, streamer=None,
            **pipeline_args):
        self.pipeline_args = pipeline_args
        self.pipeline_args.update(dict(task=task, model=model, trust_remote_code=trust_remote_code,
            device_map=device_map,
            tokenizer=tokenizer, return_full_text=return_full_text, streamer=streamer))

    def tokenizer(self) -> AutoTokenizer:
        tokenizer = self.pipeline_args["tokenizer"]
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.pipeline_args["model"])
            self.pipeline_args["tokenizer"] = tokenizer
        return tokenizer

    def create_pipeline(self, **kwargs) -> Pipeline:
        self.pipeline_args.update(kwargs)
        pipe = pipeline(**self.pipeline_args)
        return pipe

    def create_llm(self, pipeline=None, **kwargs) -> HuggingFacePipeline:
        if pipeline is None:
            pipeline = self.create_pipeline()
        return HuggingFacePipeline(pipeline=pipeline, **kwargs)

    def create_streaming_llm(self) -> StreamingLLM:
        return StreamingLLMHuggingFace(self, prompt_factory=self.create_prompt_factory())


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
        model_id = "HuggingFaceH4/starchat-alpha"
        tokenizer = AutoTokenizer.from_pretrained(model_id, eos_token="<|end|>")
        generation_config = GenerationConfig.from_pretrained(model_id)
        generation_config.eos_token_id = tokenizer.eos_token_id
        super().__init__(model_id, max_new_tokens=1024, tokenizer=tokenizer, generation_config=generation_config,
            torch_dtype=torch.bfloat16)

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


class StreamingLLMLangChainCallback(StreamingLLM):
    """
    Implements streaming via the callback mechanism that is implemented in langchain.
    At present, only OpenAI and Anthropic models support it.
    """
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


class StreamingLLMHuggingFace(StreamingLLM):
    """
    Implements streaming via a streamer that can be added to a pipeline from HuggingFace's `transformers` library.

    NOTE: Streaming will not work correctly when using the same instance for multiple parallel requests.
    """
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
        """
        :return: the chunk size to use for question answering use cases, where documents are split into chunks,
            some of which are ultimately provided as context for the question at hand.

            NOTE: The chunk size should be such that the context length is not exceeded when passing N chunks along
            with a question to the model. It cannot easily be computed, because the chunk size is in characters
            whereas the context length is in tokens (and we do not know the length, in tokens, of the question).
            So this is just a (very) rough heuristic estimate.
        """
        if self in (self.OPENAI_DAVINCI3, self.OPENAI_CHAT_GPT4):
            return 1000
        elif self in (self.HUGGINGFACE_GPT2, self.HUGGINGFACE_DOLLY_7B):
            return 512
        else:
            raise ValueError(self)
