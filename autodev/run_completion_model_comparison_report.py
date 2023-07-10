"""
Regenerates the completion model comparison report based on all results currently stored in the results folder
(as computed by the comparison script)
"""

import logging
import sys
from pathlib import Path

from autodev.autocomplete.completion_report import CompletionsHtmlDocument

log = logging.getLogger(__name__)


def create_html_for_lang(lang_id):
    results_path = Path("results") / "completion-tasks" / lang_id
    html = CompletionsHtmlDocument.from_directory(results_path, f"Example completions for language '{lang_id}'")
    html.write_html(results_path / f"results.html")


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    log.info("Starting")
    create_html_for_lang("ruby")
    log.info("Done")
