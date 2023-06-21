import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


class CompletionTask:
    TAG_COMPLETION_PLACEHOLDER = "<todo>"
    re_todo_tag = re.compile(re.escape(TAG_COMPLETION_PLACEHOLDER))

    def __init__(self, prefix: str, suffix: str, lang_id: str):
        self.prefix = prefix
        self.suffix = suffix
        self.lang_id = lang_id

    @classmethod
    def from_code_with_todo_tag(cls, code_with_todo: str, lang_id: str) -> "CompletionTask":
        m = cls.re_todo_tag.search(code_with_todo)
        assert m is not None
        prefix = code_with_todo[:m.start()]
        suffix = code_with_todo[m.end():]
        return cls(prefix, suffix, lang_id)

    def code_with_todo_tag(self) -> str:
        return self.prefix + self.TAG_COMPLETION_PLACEHOLDER + self.suffix


@dataclass
class CompletionResult:
    completion: str
    task: CompletionTask

    def full_code(self) -> str:
        return self.task.prefix + self.completion + self.task.suffix
