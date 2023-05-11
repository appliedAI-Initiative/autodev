from langchain.llms import BaseLLM

from autodev.llm import TextInTextOut


class PotentialProblemsFunction:
    PROMPT_TEMPLATE = "What are potential problems in this piece of code?\n\n" \
                      "{code}\n\n" \
                      "Potential problems:"

    def __init__(self, llm: BaseLLM):
        self.tito = TextInTextOut(llm)

    def apply(self, code: str) -> str:
        return self.tito.query(self.PROMPT_TEMPLATE.format(code=code))