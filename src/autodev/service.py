import re

from flask import Flask, request

from autodev.functions.code_function import CodeFunction, ReviewFunction, ImproveCodeFunction, ExplainCodeFunction, \
    ImplementTestsFunction, AddDocstringsFunction, PotentialProblemsFunction
from autodev.llm import LLMType


class Service:
    def __init__(self, llm_type: LLMType):
        self.app = Flask("AutoDev")
        self.llm = llm = llm_type.create_llm()
        self._add_code_function("/fn/add-comments", AddDocstringsFunction(llm))
        self._add_code_function("/fn/potential-problems", PotentialProblemsFunction(llm), html=True)
        self._add_code_function("/fn/review", ReviewFunction(llm), html=True)
        self._add_code_function("/fn/improve-code", ImproveCodeFunction(llm))
        self._add_code_function("/fn/explain", ExplainCodeFunction(llm), html=True)
        self._add_code_function("/fn/implement-tests", ImplementTestsFunction(llm))

    @staticmethod
    def _issue_code_request(fn: CodeFunction):
        print(request)
        print(request.form)
        code = request.form.get("code")
        result = fn.apply(code)
        return result

    @staticmethod
    def _format_html(response: str) -> str:
        code_blocks = []

        def extract_code(m: re.Match) -> str:
            idx = len(code_blocks)
            code_blocks.append(m.group(1))
            return f"::<-code->::{idx}"

        s = re.sub(r'```python([\s\S]*?)```', extract_code, response)

        # handle paragraphs and line breaks
        s = "<p>" + s.replace("\n\n", "</p><p>") + "</p>"
        s = s.replace("\n", "<br/>")

        def replace_code(m: re.Match) -> str:
            idx = int(m.group(1))
            code = code_blocks[idx]
            return f"<pre>{code}</pre>"

        # re-insert code blocks
        s = re.sub(r"::<-code->::(\d+)", replace_code, s)

        s = '<div style="font-family: monospace">' + s + "</div>"
        return s

    def _add_code_function(self, path, fn: CodeFunction, html=False):
        def handle():
            response = self._issue_code_request(fn)
            if html:
                response = self._format_html(response)
            return response
        handle.__name__ = fn.__class__.__name__

        self.app.add_url_rule(path, None, handle, methods=["POST"])

    def run(self):
        self.app.run()


if __name__ == '__main__':
    Service(LLMType.OPENAI_CHAT_GPT4).run()