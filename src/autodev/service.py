import re
import sys

from flask import Flask, request

from autodev.functions.code_function import CodeFunction, ReviewFunction, ImproveCodeFunction, ExplainCodeFunction, \
    ImplementTestsFunction, AddDocstringsFunction, PotentialProblemsFunction
from autodev.llm import LLMType
from autodev.response_stream_formatting import StreamHtmlFormatter


class Service:
    def __init__(self, llm_type: LLMType):
        self.app = Flask("AutoDev")
        self.sllm = llm = llm_type.create_streaming_llm()
        self._add_code_function("/fn/add-comments", AddDocstringsFunction(llm))
        self._add_code_function("/fn/potential-problems", PotentialProblemsFunction(llm), html=True)
        self._add_code_function("/fn/review", ReviewFunction(llm), html=True)
        self._add_code_function("/fn/improve-code", ImproveCodeFunction(llm))
        self._add_code_function("/fn/explain", ExplainCodeFunction(llm), html=True)
        self._add_code_function("/fn/implement-tests", ImplementTestsFunction(llm))
        self._add_streaming_code_function("/fn/stream/potential-problems", PotentialProblemsFunction(llm), html=True)
        self._add_streaming_code_function("/fn/stream/review", ReviewFunction(llm), html=True)
        self._add_streaming_code_function("/fn/stream/explain", ExplainCodeFunction(llm), html=True)
        self._add_streaming_code_function("/fn/stream/implement-tests", ImplementTestsFunction(llm), html=False)

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
            print(request)
            print(request.form)
            code = request.form.get("code")
            response = fn.apply(code)
            if html:
                response = self._format_html(response)
            return response

        handle.__name__ = fn.__class__.__name__
        self.app.add_url_rule(path, None, handle, methods=["POST"])

    def _add_streaming_code_function(self, path, fn: CodeFunction, html=True):
        def handle():
            print(request)
            print(request.form)
            code = request.form.get("code")

            def generate_plain():
                for token in fn.apply_streaming(code):
                    sys.stdout.write(token)
                    yield token

            def generate_html():
                formatter = StreamHtmlFormatter()

                for token in fn.apply_streaming(code):
                    sys.stdout.write(token)
                    yield from formatter.append(token)

                yield formatter.flush()

            if html:
                generate = generate_html
                mimetype = "text/html"
            else:
                generate = generate_plain
                mimetype = "text/plain"
            return self.app.response_class(generate(), mimetype=mimetype)

        handle.__name__ = "stream_" + fn.__class__.__name__
        return self.app.add_url_rule(path, None, handle, methods=["POST"])

    def run(self):
        self.app.run()
