"""
Supports the formatting of streamed LLM responses, particularly for the generation of HTML output
"""
import logging
import re
from abc import abstractmethod, ABC
from enum import Enum
from typing import Optional, Tuple, Iterator

log = logging.getLogger(__name__)


class FormatterRuleStatus(Enum):
    """
    Status of formatter rule application
    """
    INAPPLICABLE = 0  # rule does not apply
    APPLIED = 1  # rule was applied (buffer was modified, HTML output was generated)
    WITHHELD = 2  # rule applicability cannot be determined (extended buffer needed)


class State(Enum):
    """
    State of the formatter indicating the type of content currently being processed
    """
    TEXT = 0
    CODE = 1


class FormatterRule(ABC):
    def __init__(self, sf: "StreamHtmlFormatter",
            init_char: str,
            rule_applies_regex: str,
            rule_void_regex: Optional[str],
            state: "State"):
        """
        :param sf: the stream formatter whose data shall be affected by this rule
        :param init_char: the initial character(s) with which a buffer must start for this rule to potentially be
            applicable
        :param rule_applies_regex: the regex that must match the beginning of the buffer for this rule to be
            applicable (the match must begin with `init_char`)
        :param rule_void_regex: the regex which, if matched, indicates that this rule is not applicable
        :param state: the formatter state in which this rule applies
        """
        self.state = state
        self.sf = sf
        self.init_char = init_char
        self.rule_applies_regex = re.compile(rule_applies_regex)
        self.rule_void_regex = re.compile(rule_void_regex) if rule_void_regex else None

    def check(self) -> Tuple[FormatterRuleStatus, Optional[str]]:
        """
        Checks whether the rule is applicable and if so, modifies the StreamHtmlFormatter instance's
        buffer and returns the HTML output to append as a second element

        :return: a tuple containing the application status and the HTML output to append
        """
        if self.sf.state != self.state:
            return FormatterRuleStatus.INAPPLICABLE, None
        buf = self.sf.buffer
        if buf[0] not in self.init_char:
            return FormatterRuleStatus.INAPPLICABLE, None
        m = self.rule_applies_regex.match(buf)
        if m:
            html, buf = self._apply_format(m, buf)
            self.sf.buffer = buf
            return FormatterRuleStatus.APPLIED, html
        elif self.rule_void_regex:
            m = self.rule_void_regex.match(buf)
            if m:
                return FormatterRuleStatus.INAPPLICABLE, None
        return FormatterRuleStatus.WITHHELD, None

    @abstractmethod
    def _apply_format(self, m: re.Match, s: str) -> Tuple[str, str]:
        """
        Applies the rule given the match `m` at the beginning of the buffer `s`

        :param m: a regex Match object (which matches this rule's applicability regex)
        :param s: the buffer at the beginning of which the match was found
        :return: a pair (html_output, buffer) where html_output is the HTML output to add
            and buffer is the new (reduced) buffer after applying the rule
        """
        pass


class FormatterRuleCodeBlockStart(FormatterRule):
    def __init__(self, sf: "StreamHtmlFormatter"):
        super().__init__(sf, "\n \t", r"(\s```(\w+)?)\s", r"\s([^`]|`[^`]|``[^`]|```\W)", State.TEXT)

    def _apply_format(self, m: re.Match, s: str) -> Tuple[str, str]:
        self.sf.state = State.CODE
        return "\n<pre>", s[len(m.group(1)):]


class FormatterRuleCodeBlockEnd(FormatterRule):
    def __init__(self, sf: "StreamHtmlFormatter"):
        super().__init__(sf, "\n \t", r"(\s```)", r"\s([^`]|`[^`]|``[^`])", State.CODE)

    def _apply_format(self, m: re.Match, s: str) -> Tuple[str, str]:
        self.sf.state = State.TEXT
        return "\n</pre>", s[len(m.group(1)):]


class FormatterRuleNl2Br(FormatterRule):
    def __init__(self, sf: "StreamHtmlFormatter"):
        super().__init__(sf, "\n", r"\n", None, State.TEXT)

    def _apply_format(self, m: re.Match, s: str) -> Tuple[str, str]:
        return "<br>\n", s[1:]


class StreamHtmlFormatter:
    def __init__(self):
        self.rules = [
            FormatterRuleCodeBlockStart(self),
            FormatterRuleCodeBlockEnd(self),
            FormatterRuleNl2Br(self),
        ]
        self.html_output = ""
        self.buffer = ""
        self.state = State.TEXT

    def append(self, s: str) -> Iterator[str]:
        for c in s:
            yield from self._append(c)

    def _append(self, s: str) -> Iterator[str]:
        self.buffer += s

        done = False
        while self.buffer != "" and not done:
            log.debug(f"buf=\"{self.buffer}\"")
            for rule in self.rules:
                status, html = rule.check()
                if html is not None:
                    self.html_output += html
                    yield html
                log.debug(f"rule {rule}: {status}")
                if status == FormatterRuleStatus.WITHHELD:
                    done = True  # need greater buffer to determine rule applicability
                    break
                elif status == FormatterRuleStatus.APPLIED:
                    break  # restart rule checks with modified buffer
            else:  # no rule returned positive, so move one char to the HTML output
                yield self.flush(1)

    def flush(self, n: Optional[int] = None):
        """
        Flushes the buffer to the HTML output.

        :param n: the number of characters to flush; if None, flushes the entire buffer
        :return: the HTML output that was flushed
        """
        if n is None:
            n = len(self.buffer)
        html_added = self.buffer[:n]
        self.html_output += html_added
        self.buffer = self.buffer[n:]
        return html_added
