import logging
import os
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import pipeline

from completionft.model import model_id_to_fn, ModelFactory

log = logging.getLogger(__name__)


TAG_COMPLETION_PLACEHOLDER = "<todo>"
TAG_FIM_PREFIX = "<fim-prefix>"
TAG_FIM_SUFFIX = "<fim-suffix>"
TAG_FIM_MIDDLE = "<fim-middle>"

re_todo_tag = re.compile(re.escape(TAG_COMPLETION_PLACEHOLDER))
re_fim_middle = re.compile(re.escape(TAG_FIM_MIDDLE))


class CompletionTask:
    def __init__(self, prefix: str, suffix: str, lang_id: str):
        self.prefix = prefix
        self.suffix = suffix
        self.middle = None
        self.lang_id = lang_id

    @classmethod
    def from_code_with_todo_tag(cls, code_with_todo: str, lang_id: str) -> "CompletionTask":
        m = re_todo_tag.search(code_with_todo)
        assert m is not None
        prefix = code_with_todo[:m.start()]
        suffix = code_with_todo[m.end():]
        return cls(prefix, suffix, lang_id)

    def fim_prompt(self) -> str:
        return f"{TAG_FIM_PREFIX}{self.prefix}{TAG_FIM_SUFFIX}{self.suffix}{TAG_FIM_MIDDLE}"

    def _extract_completion(self, s: str) -> str:
        m = re_fim_middle.search(s)
        if not m:
            return ""
        completion = s[m.end():]

        # truncate completion depending on lang_id
        if self.lang_id == "ruby":  # end completion after first occurrence of 'end'
            m = re.search(r"end\n", completion)
            if m:
                completion = completion[:m.end()]
                if self.suffix.startswith("\n"):
                    completion = completion[:-1]

        return completion

    def apply_completion(self, model_response: str) -> None:
        self.middle = self._extract_completion(model_response)

    def full_code(self) -> str:
        return self.prefix + self.middle + self.suffix

    def code_with_todo_tag(self) -> str:
        return self.prefix + TAG_COMPLETION_PLACEHOLDER + self.suffix


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

    def run(self, save_results=True):
        tasks = self.completion_tasks
        tokenizer = self.model_factory.create_tokenizer()

        def result_dir(task_name):
            outdir = Path("results") / "completion-tasks" / self.lang_id / task_name
            outdir.mkdir(parents=True, exist_ok=True)
            return outdir

        for model_id in self.model_paths:

            log.info(f"Loading model {model_id}")
            model = self.model_factory.create_model(model_id)
            pipe = pipeline("text-generation", model=model, max_new_tokens=256, device=self.device,
                torch_dtype=torch.bfloat16, trust_remote_code=True, tokenizer=tokenizer)

            for task_name, task in tasks.items():
                ext = os.path.splitext(task_name)[1]
                if save_results:
                    with open(result_dir(task_name) / f"task{ext}", 'w') as f:
                        f.write(task.code_with_todo_tag())

                log.info(f"Querying {model_id} for completion {task_name}")
                prompt = task.fim_prompt()
                response = pipe(prompt)[0]["generated_text"]
                task.apply_completion(response)
                log.info(f"Completion for {task_name} by {model_id}:\n{task.full_code()}")

                if save_results:
                    fn = model_id_to_fn(model_id) + ext
                    with open(result_dir(task_name) / fn, 'w') as f:
                        f.write(task.full_code())

            del model
            del pipe
