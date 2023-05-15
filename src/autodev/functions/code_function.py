"""
LLM functions that take code as input and return text/code.
"""

from abc import ABC, abstractmethod

from langchain.llms import BaseLLM

from autodev.llm import TextInTextOut


class CodeFunction(ABC):
    def __init__(self, llm: BaseLLM):
        self.tito = TextInTextOut(llm)

    @abstractmethod
    def apply(self, code: str) -> str:
        pass


class AddDocstringsFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please add docstrings to this piece of code:\n\n" \
        "{code}\n\n" \
        "Commented code:"

    def apply(self, code: str) -> str:
        return self.tito.query(self.PROMPT_TEMPLATE.format(code=code))


class ExplainCodeFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please describe what this piece of code does:\n\n" \
        "{code}"

    def apply(self, code: str) -> str:
        return self.tito.query(self.PROMPT_TEMPLATE.format(code=code))


class ImplementTestsFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please implement tests for this piece of code:\n\n" \
        "{code}\n\n" \
        "Test code:"

    def apply(self, code: str) -> str:
        return "<pre>" + self.tito.query(self.PROMPT_TEMPLATE.format(code=code)) + "</pre>"


class ImproveCodeFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please identify flaws in this piece of code and respond with an improved version," \
        "explaining your improvements in code comments:\n\n" \
        "{code}\n\n" \
        "Improved code:"

    def apply(self, code: str) -> str:
        return self.tito.query(self.PROMPT_TEMPLATE.format(code=code))


class PotentialProblemsFunction(CodeFunction):
    PROMPT_TEMPLATE = "What are potential problems in this piece of code?\n\n" \
        "{code}\n\n" \
        "Potential problems:"

    def apply(self, code: str) -> str:
        return self.tito.query(self.PROMPT_TEMPLATE.format(code=code))


class ReviewFunction(CodeFunction):
    PROMPT_TEMPLATE = "Please review this piece of code:\n\n{code}"

    def apply(self, code: str) -> str:
        return self.tito.query(self.PROMPT_TEMPLATE.format(code=code))