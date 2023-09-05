from transformers import PreTrainedTokenizer


class FIMTokenIds:
    def __init__(self, prefix_token_id: int, suffix_token_id: int, middle_token_id: int, pad_token_id: int):
        self.prefix_token_id = prefix_token_id
        self.suffix_token_id = suffix_token_id
        self.middle_token_id = middle_token_id
        self.pad_token_id = pad_token_id


class FIMTokens:
    def __init__(self, prefix_token: str, suffix_token: str, middle_token: str, pad_token: str):
        self.prefix_token = prefix_token
        self.suffix_token = suffix_token
        self.middle_token = middle_token
        self.pad_token = pad_token

    def get_token_ids(self, tokenizer: PreTrainedTokenizer) -> FIMTokenIds:
        prefix_token_id = self._get_token_id(self.prefix_token, tokenizer)
        suffix_token_id = self._get_token_id(self.suffix_token, tokenizer)
        middle_token_id = self._get_token_id(self.middle_token, tokenizer)
        pad_token_id = self._get_token_id(self.pad_token, tokenizer)
        return FIMTokenIds(prefix_token_id, suffix_token_id, middle_token_id, pad_token_id)

    def _get_token_id(self, token: str, tokenizer: PreTrainedTokenizer):
        return tokenizer.vocab[token]


class BigCodeFIMTokens(FIMTokens):
    def __init__(self):
        super().__init__("<fim-prefix>", "<fim-suffix>", "<fim-middle>", "<fim-pad>")
