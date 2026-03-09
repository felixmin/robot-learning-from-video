from __future__ import annotations

import pytest

from stage2.action_tokens import (
    ActionTokenConfig,
    add_action_tokens,
    allowed_action_token_ids,
    extract_code_token_ids,
)


class DummyTokenizer:
    """
    Minimal tokenizer stub that mimics the two Hugging Face methods we rely on.
    """

    def __init__(self):
        self._token_to_id: dict[str, int] = {}
        self._next_id = 1

    def add_special_tokens(self, spec):
        tokens = spec.get("additional_special_tokens", [])
        added = 0
        for tok in tokens:
            if tok not in self._token_to_id:
                self._token_to_id[tok] = self._next_id
                self._next_id += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, -1)


def test_format_target():
    cfg = ActionTokenConfig(codebook_size=8, code_seq_len=4)
    assert (
        cfg.format_target([3, 1, 7, 0])
        == "<ACTION> <ACT_3> <ACT_1> <ACT_7> <ACT_0> </ACTION>"
    )


def test_validate_codes_length():
    cfg = ActionTokenConfig(codebook_size=8, code_seq_len=4)
    with pytest.raises(ValueError, match="Expected 4 codes"):
        cfg.format_target([0, 1, 2])


def test_validate_codes_range():
    cfg = ActionTokenConfig(codebook_size=8, code_seq_len=4)
    with pytest.raises(ValueError, match="out of range"):
        cfg.format_target([0, 1, 2, 9])


def test_add_action_tokens_and_ids():
    cfg = ActionTokenConfig(codebook_size=8, code_seq_len=4)
    tok = DummyTokenizer()
    token_id_map = add_action_tokens(tok, cfg)

    assert set(token_id_map.keys()) == set(cfg.all_tokens())
    assert all(i > 0 for i in token_id_map.values())

    allowed = allowed_action_token_ids(token_id_map, cfg)
    assert len(allowed) == len(cfg.all_tokens())
    assert len(set(allowed)) == len(allowed)


def test_extract_code_token_ids_excludes_wrappers():
    cfg = ActionTokenConfig(codebook_size=8, code_seq_len=4)
    tok = DummyTokenizer()
    token_id_map = add_action_tokens(tok, cfg)

    action_start = token_id_map[cfg.action_start]
    action_end = token_id_map[cfg.action_end]
    code_ids = [token_id_map[f"<ACT_{i}>"] for i in [3, 1, 7, 0]]

    seq = [111, action_start, *code_ids, action_end, 222]
    assert extract_code_token_ids(seq, token_id_map, cfg) == code_ids
