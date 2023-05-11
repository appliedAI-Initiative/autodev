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


@app.route('/fn/add-comments', methods=['POST'])
def add_comments():
    return issue_code_request(fn_add_comments)


@app.route('/fn/potential-problems', methods=['POST'])
def potential_problems():
    return issue_code_request(fn_potential_problems)


if __name__ == '__main__':
    app.run()