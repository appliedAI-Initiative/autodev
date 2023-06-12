import logging
import re

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
