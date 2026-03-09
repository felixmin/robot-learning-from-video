from __future__ import annotations

from stage2.action_tokens import ActionTokenConfig
from stage2.legacy.qwen3vl_setup import prepare_action_token_training


class DummyTokenizer:
    def __init__(self):
        self._tokens = {}

    def add_special_tokens(self, spec):
        for tok in spec.get("additional_special_tokens", []):
            self._tokens.setdefault(tok, len(self._tokens) + 1)
        return 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens.get(token, -1)

    def __len__(self):
        return len(self._tokens)


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


class DummyModel:
    def __init__(self):
        self.last_resize = None

    def resize_token_embeddings(self, n: int):
        self.last_resize = n


def test_prepare_action_token_training_adds_tokens_and_resizes():
    processor = DummyProcessor()
    model = DummyModel()
    cfg = ActionTokenConfig(codebook_size=8, code_seq_len=4)

    token_id_map = prepare_action_token_training(
        model=model, processor=processor, action_tokens=cfg
    )

    assert set(token_id_map.keys()) == set(cfg.all_tokens())
    assert model.last_resize == len(processor.tokenizer)
