import collections
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from completionft.completion_model import CompletionModel
from completionft.completion_report import CompletionsHtmlDocument
from completionft.completion_task import CompletionTask
from completionft.model import model_id_to_fn, ModelFactory

log = logging.getLogger(__name__)


class CompletionTaskModelComparison:
    def __init__(self,
            lang_id: str,
            model_factory: ModelFactory,
            model_paths: List[str],
            device="cuda:0"):
        """
        :param lang_id: the language id for which to load the completion tasks
        :param model_factory: the factory with which to create models
        :param model_paths: paths with which to call model_factory in order to obtain the concrete models
        :param device: the torch device to use
        """
        self.lang_id = lang_id
        self.completion_tasks = self._read_completion_tasks(lang_id)
        self.model_factory = model_factory
        self.device = device
        self.model_paths = model_paths

    @staticmethod
    def _read_completion_tasks(lang_id) -> Dict[str, CompletionTask]:
        tasks = {}
        root = Path("data") / "completion-tasks" / lang_id
        for fn in os.listdir(root):
            with open(root / fn, "r") as f:
                code = f.read()
                tasks[fn] = CompletionTask.from_code_with_todo_tag(code, lang_id)
        return tasks

    def run(self, save_results=True, save_html_report=True):
        tasks = self.completion_tasks
        tokenizer = self.model_factory.create_tokenizer()

        def result_dir(task_name):
            outdir = Path("results") / "completion-tasks" / self.lang_id / task_name
            outdir.mkdir(parents=True, exist_ok=True)
            return outdir

        completions = collections.defaultdict(dict)
        for model_id in self.model_paths:

            log.info(f"Loading model {model_id}")
            model = self.model_factory.create_model(model_id)
            completion_model = CompletionModel(model, tokenizer, device=self.device)

            for task_name, task in tasks.items():
                ext = os.path.splitext(task_name)[1]
                if save_results:
                    with open(result_dir(task_name) / f"task{ext}", 'w') as f:
                        f.write(task.code_with_todo_tag())

                log.info(f"Querying {model_id} for completion {task_name}")
                result = completion_model.apply(task)
                completed_code = result.full_code()
                log.info(f"Completion for {task_name} by {model_id}:\n{completed_code}")
                completions[task_name][model_id] = completed_code

                if save_results:
                    fn = model_id_to_fn(model_id) + ext
                    with open(result_dir(task_name) / fn, 'w') as f:
                        f.write(completed_code)

            del model
            del completion_model

        if save_html_report:
            html = CompletionsHtmlDocument(self.lang_id)
            for task_name, task in tasks.items():
                html.add_task(task_name, task.code_with_todo_tag(), completions[task_name])
            html.finish()
            outdir = Path("results") / "completion-tasks" / self.lang_id
            outdir.mkdir(parents=True, exist_ok=True)
            tag = datetime.now().strftime('%Y%m%d-%H%M%S')
            html.write_html(outdir / f"results-{tag}.html")