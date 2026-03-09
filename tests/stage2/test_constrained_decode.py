from __future__ import annotations

from stage2.constrained_decode import ActionTokenIds, make_prefix_allowed_tokens_fn


def test_constrained_decode_state_machine():
    ids = ActionTokenIds(
        action_start_id=10,
        action_end_id=11,
        action_code_ids=list(range(20, 28)),
        between_token_ids=[99],
        eos_token_id=2,
        code_seq_len=4,
    )

    # Before start: must emit <ACTION>
    assert ids.next_allowed_ids([101, 102]) == [10]

    # After <ACTION>: can emit any code token until length is satisfied
    assert ids.next_allowed_ids([101, 10]) == [99] + list(range(20, 28))
    # After a separator, must emit a code (avoid infinite separators).
    assert ids.next_allowed_ids([101, 10, 99]) == list(range(20, 28))
    assert ids.next_allowed_ids([101, 10, 99, 20, 99, 21]) == [99] + list(range(20, 28))

    # After 4 codes: must emit </ACTION>
    assert ids.next_allowed_ids([101, 10, 99, 20, 99, 21, 99, 22, 99, 23]) == [99, 11]
    # After the separator before </ACTION>, force </ACTION>.
    assert ids.next_allowed_ids([101, 10, 99, 20, 99, 21, 99, 22, 99, 23, 99]) == [11]

    # After </ACTION>: must emit EOS
    assert ids.next_allowed_ids([101, 10, 20, 21, 22, 23, 11]) == [2]


def test_prefix_allowed_tokens_fn_signature():
    ids = ActionTokenIds(
        action_start_id=10,
        action_end_id=11,
        action_code_ids=list(range(20, 28)),
        between_token_ids=[99],
        eos_token_id=2,
        code_seq_len=4,
    )
    fn = make_prefix_allowed_tokens_fn(ids)

    class _T:
        def __init__(self, xs):
            self._xs = xs

        def tolist(self):
            return list(self._xs)

    assert fn(0, _T([1, 2, 3])) == [10]
