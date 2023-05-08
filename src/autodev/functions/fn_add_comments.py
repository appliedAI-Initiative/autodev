from langchain.llms import BaseLLM

from autodev.llm import TextInTextOut


class AddCommentsFunction:
    PROMPT_TEMPLATE = "Please add docstrings to this piece of code:\n\n" \
        "{code}\n\n" \
        "Commented code:"

    def __init__(self, llm: BaseLLM):
        self.tito = TextInTextOut(llm)

    def apply(self, code: str) -> str:
        return self.tito.query(self.PROMPT_TEMPLATE.format(code=code))