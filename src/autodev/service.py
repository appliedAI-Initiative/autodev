from flask import Flask, request

from autodev.functions.fn_add_comments import AddCommentsFunction
from autodev.llm import LLMType

app = Flask(__name__)
llm = LLMType.OPENAI_CHAT_GPT4.create_llm()

fn_add_comments = AddCommentsFunction(llm)


@app.route('/fn/add-comments', methods=['POST'])
def add_comments():
    print(request)
    print(request.form)
    code = request.form.get("code")
    result = fn_add_comments.apply(code)
    return result


if __name__ == '__main__':
    app.run()