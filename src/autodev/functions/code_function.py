"""
LLM functions that take code as input and return text/code.
"""

from abc import ABC, abstractmethod
from typing import Iterator

from autodev.llm import StreamingLLM


class CodeFunction(ABC):
    def __init__(self, sllm: StreamingLLM):
        self.sllm = sllm

    @abstractmethod
    def generate_prompt(self, code: str) -> str:
        pass

    def apply(self, code: str, stream=None) -> str:
        prompt = self.generate_prompt(code)
        return self.sllm.query(prompt, stream=stream)

    def apply_streaming(self, code: str) -> Iterator[str]:
        prompt = self.generate_prompt(code)
        yield from self.sllm.query_streaming(prompt)


class AddDocstringsFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please add docstrings to this piece of code:\n\n" \
        "{code}\n\n" \
        "Commented code:"

    def generate_prompt(self, code: str) -> str:
        return self.PROMPT_TEMPLATE.format(code=code)


class ExplainCodeFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please describe what this piece of code does:\n\n" \
        "{code}"

    def generate_prompt(self, code: str) -> str:
        return self.PROMPT_TEMPLATE.format(code=code)


class ImplementTestsFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please implement tests for this piece of code:\n\n" \
        "{code}\n\n" \
        "Test code:"

    def generate_prompt(self, code: str) -> str:
        return self.PROMPT_TEMPLATE.format(code=code)


class ImproveCodeFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please identify flaws in this piece of code and respond with an improved version," \
        "explaining your improvements in code comments:\n\n" \
        "{code}\n\n" \
        "Improved code:"

    def generate_prompt(self, code: str) -> str:
        return self.PROMPT_TEMPLATE.format(code=code)


class PotentialProblemsFunction(CodeFunction):
    PROMPT_TEMPLATE = "What are potential problems in this piece of code?\n\n" \
        "{code}\n\n" \
        "Potential problems:"

    def generate_prompt(self, code: str) -> str:
        return self.PROMPT_TEMPLATE.format(code=code)


class ReviewFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please review this piece of code:\n\n{code}"

    def generate_prompt(self, code: str) -> str:
        return self.PROMPT_TEMPLATE.format(code=code)