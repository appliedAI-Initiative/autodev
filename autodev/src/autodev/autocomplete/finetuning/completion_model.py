import re
from typing import Union

import torch
from peft import PeftModel
from transformers import PreTrainedModel, pipeline

from completionft.completion_task import CompletionTask, CompletionResult


class CompletionModel:
    TAG_FIM_PREFIX = "<fim-prefix>"
    TAG_FIM_SUFFIX = "<fim-suffix>"
    TAG_FIM_MIDDLE = "<fim-middle>"
    re_fim_middle = re.compile(re.escape(TAG_FIM_MIDDLE))

    def __init__(self,
            model: Union[PreTrainedModel, PeftModel],
            tokenizer,
            max_new_tokens=256,
            torch_dtype=torch.bfloat16,
            device="cuda:0"):
        self.model = model
        self.pipe = pipeline("text-generation", model=model, max_new_tokens=max_new_tokens, device=device,
            torch_dtype=torch_dtype, trust_remote_code=True, tokenizer=tokenizer)

    def fim_prompt(self, task: CompletionTask) -> str:
        return f"{self.TAG_FIM_PREFIX}{task.prefix}{self.TAG_FIM_SUFFIX}{task.suffix}{self.TAG_FIM_MIDDLE}"

    def _extract_completion(self, s: str, task: CompletionTask) -> str:
        m = self.re_fim_middle.search(s)
        if not m:
            return ""
        completion = s[m.end():]

        # truncate completion depending on lang_id
        if task.lang_id == "ruby":
            # if next line of suffix is an 'end' line, end completion before respective 'end'
            suffix_lines = task.suffix.split("\n")
            if len(suffix_lines) >= 2 and suffix_lines[1].strip() == "end":
                end_line = suffix_lines[1].rstrip() + "\n"
                m = re.search(r"^" + re.escape(end_line), completion, re.MULTILINE)
                if m:
                    completion = completion[:m.start()-1]

        return completion

    def apply(self, task: CompletionTask) -> CompletionResult:
        prompt = self.fim_prompt(task)
        response = self.pipe(prompt)[0]["generated_text"]
        completion = self._extract_completion(response, task)
        return CompletionResult(completion, task)

