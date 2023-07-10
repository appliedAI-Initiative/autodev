"""
This module contains code for the generation of HTML files that provide a report of completions
generated by various models on the same tasks - to facilitate qualitative analysis based on examples.
"""

from html import escape
import logging
import os
from pathlib import Path

from completionft.completion_task import CompletionTask
from completionft.model import model_id_from_fn

log = logging.getLogger(__name__)


HTML_PREFIX = """<html>
<head>
    <style>
        body {
            font-family: sans-serif;
            padding: 30px;
        }
        .code {
            white-space: pre;
            font-family: Courier New, monospace;
            background-color: #eee;
            margin-top: 10px;
            padding: 10px;
        }
        .placeholder {
            color: red;
        }
        .task {
        }
        table.completions {
            width: 100%;
        }
        table.completions .code {
            overflow-x: scroll;
        }
        table.completions td {
            vertical-align: top;
            padding: 10px;
        }
        table.completions td:first-child {
            padding-left: 0;
        }
    </style>
</head>
<body>"""

HTML_SUFFIX = "</body></html>"


class CompletionsHtmlDocument:
    ESCAPED_TAG_COMPLETION_PLACEHOLDER = escape(CompletionTask.TAG_COMPLETION_PLACEHOLDER)

    def __init__(self, title):
        self.html = HTML_PREFIX
        self.html += f"<h1>{title}</h1>"

    def add_task(self, task_name, task_description: str, completions: dict[str, str]):
        self.html += f"<h2>{task_name}</h2>"

        if CompletionsHtmlDocument.ESCAPED_TAG_COMPLETION_PLACEHOLDER in task_description:
            task_description = task_description.replace(CompletionsHtmlDocument.ESCAPED_TAG_COMPLETION_PLACEHOLDER,
                f'<span class="placeholder">{CompletionsHtmlDocument.ESCAPED_TAG_COMPLETION_PLACEHOLDER}</span>')
        self.html += f'<h3>Task</h3><div class="code task">{task_description}</div>'

        self.html += '<h3>Completions</h3><table class="completions"><tr>'
        columns = 2
        for i, model_name in enumerate(sorted(completions.keys()), start=1):
            completion = completions[model_name]
            self.html += f'<td style="">{model_name}<br><div class="code">{completion}</td>'
            if i % columns == 0:
                self.html += "</tr>"
                if len(completions) > i:
                    self.html += "<tr>"
        self.html += "</tr></table>"

    def finish(self):
        self.html += HTML_SUFFIX

    def write_html(self, path):
        log.info(f"Writing HTML output to {path}")
        with open(path, "w") as f:
            f.write(self.html)

    @staticmethod
    def from_directory(root_dir: Path, title: str) -> "CompletionsHtmlDocument":
        """
        Generations an HTML document with completions from a directory containing stored completions
            (as generated by class CompletionTaskModelComparison),
            with each set of completions in a separate directory named according to the task's name.
            Each directory is to contain one file named task.txt with the task description as well as
            several files containing completions, named after the model that generated the completion.

        :param root_dir: the directory
        :param title: the title of the document
        :return: the HTML document
        """
        html = CompletionsHtmlDocument(title)

        for task_name in os.listdir(root_dir):
            task_dir = root_dir / task_name
            if not task_dir.is_dir():
                continue

            log.info(f"Task '{task_name}'")

            completions = {}
            task_description = ""
            for fn in os.listdir(task_dir):
                content = (task_dir / fn).read_text()
                escaped_content = escape(content)
                basename = os.path.splitext(fn)[0]
                if basename == "task":
                    task_description = escaped_content
                else:
                    model_name = model_id_from_fn(basename)
                    completions[model_name] = escaped_content

            html.add_task(task_name, task_description, completions)

        html.finish()
        return html

