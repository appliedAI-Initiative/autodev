import re

from flask import Flask, request

from autodev.functions.fn_add_comments import AddDocstringsFunction
from autodev.functions.fn_potential_problems import PotentialProblemsFunction
from autodev.llm import LLMType

app = Flask(__name__)
llm = LLMType.OPENAI_CHAT_GPT4.create_llm()

fn_add_comments = AddDocstringsFunction(llm)
fn_potential_problems = PotentialProblemsFunction(llm)


def issue_code_request(fn):
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


@app.route('/fn/add-comments', methods=['POST'])
def add_comments():
    return issue_code_request(fn_add_comments)


@app.route('/fn/potential-problems', methods=['POST'])
def potential_problems():
    response = issue_code_request(fn_potential_problems)
    return format_html(response)


if __name__ == '__main__':
    app.run()