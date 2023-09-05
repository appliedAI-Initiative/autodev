import re
from typing import Union

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from peft import PeftModel
from transformers import PreTrainedModel, pipeline

from .completion_task import CompletionTask, CompletionResult
from .fim_config import FIMTokens
from .model import ModelFactory


def fim_prompt(task: CompletionTask, fim_tokens: FIMTokens) -> str:
    return f"{fim_tokens.prefix_token}{task.prefix}{fim_tokens.suffix_token}{task.suffix}{fim_tokens.middle_token}"


class CompletionModel:
    DEBUG = False

    def __init__(self,
            model: Union[PreTrainedModel, PeftModel, ORTModelForCausalLM],
            tokenizer,
            fim_tokens: FIMTokens,
            max_new_tokens=256,
            device="cuda:0"):
        self.model = model
        self.fim_tokens = fim_tokens
        self.pipe = pipeline("text-generation", model=model, max_new_tokens=max_new_tokens, device=device,
            trust_remote_code=True, tokenizer=tokenizer)
        self.re_fim_middle = re.compile(re.escape(fim_tokens.middle_token))

    @classmethod
    def from_model_factory(cls, model_factory: ModelFactory, model_path=None, max_tokens=256, device=None) -> "CompletionModel":
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return CompletionModel(model_factory.create_model(model_path),
            model_factory.create_tokenizer(),
            model_factory.fim_tokens,
            device=device,
            max_new_tokens=max_tokens)

    def fim_prompt(self, task: CompletionTask) -> str:
        return fim_prompt(task, self.fim_tokens)

    def _extract_completion(self, s: str, task: CompletionTask) -> str:
        if self.DEBUG:
            import pickle
            with open("completion.pkl", "wb") as f:
                pickle.dump({"task": task, "s": s}, f)

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

