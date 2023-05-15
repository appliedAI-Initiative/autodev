import re

from flask import Flask, request

from autodev.functions.code_function import CodeFunction, ReviewFunction, ImproveCodeFunction, ExplainCodeFunction, \
    ImplementTestsFunction, AddDocstringsFunction, PotentialProblemsFunction
from autodev.llm import LLMType


def issue_code_request(fn: CodeFunction):
    print(request)
    print(request.form)
    code = request.form.get("code")
    result = fn.apply(code)
    return result


def format_html(response: str) -> str:
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


def add_code_function(path, fn: CodeFunction, html=False):
    def handle():
        response = issue_code_request(fn)
        if html:
            response = format_html(response)
        return response
    handle.__name__ = fn.__class__.__name__

    app.add_url_rule(path, None, handle, methods=["POST"])


if __name__ == '__main__':
    app = Flask(__name__)
    llm = LLMType.OPENAI_CHAT_GPT4.create_llm()
    add_code_function("/fn/add-comments", AddDocstringsFunction(llm))
    add_code_function("/fn/potential-problems", PotentialProblemsFunction(llm), html=True)
    add_code_function("/fn/review", ReviewFunction(llm), html=True)
    add_code_function("/fn/improve-code", ImproveCodeFunction(llm))
    add_code_function("/fn/explain", ExplainCodeFunction(llm), html=True)
    add_code_function("/fn/implement-tests", ImplementTestsFunction(llm))
    app.run()