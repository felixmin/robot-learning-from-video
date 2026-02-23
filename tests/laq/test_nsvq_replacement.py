"""
Tests for NSVQ codebook replacement logic.

Focuses on the size-aware replacement criterion used by replace_unused_codebooks.
"""

import torch


def _make_nsvq(num_embeddings: int, discarding_threshold: float):
    from laq.models.nsvq import NSVQ

    vq = NSVQ(
        dim=8,
        num_embeddings=num_embeddings,
        embedding_dim=4,
        discarding_threshold=discarding_threshold,
        code_seq_len=1,
        patch_size=32,
        image_size=256,
        grid_size=(8, 8),
    )
    # Tests manually set usage counts; avoid coupling to forward().
    vq.codebooks_used.zero_()
    return vq


class TestNSVQReplacementIndices:
    def test_size_aware_threshold_only_marks_zeros_when_expected_lt_one(self):
        vq = _make_nsvq(num_embeddings=100, discarding_threshold=0.1)

        # Very few assignments in a large codebook => expected usage < 1 per entry.
        vq.codebooks_used[3] = 1
        vq.codebooks_used[7] = 1

        unused, used, min_count = vq._get_replacement_indices()

        assert min_count < 1.0
        assert set(used.tolist()) == {3, 7}
        assert 3 not in set(unused.tolist())
        assert 7 not in set(unused.tolist())

    def test_threshold_is_fraction_of_average_usage(self):
        vq = _make_nsvq(num_embeddings=10, discarding_threshold=0.5)

        # total=25 => expected=2.5 => min_count=1.25
        # used: counts >= 1.25 => {0,1,2}; unused includes index 3 (count=1)
        vq.codebooks_used[:] = torch.tensor([20, 2, 2, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)

        unused, used, min_count = vq._get_replacement_indices()

        assert abs(min_count - 1.25) < 1e-6
        assert set(used.tolist()) == {0, 1, 2}
        assert 3 in set(unused.tolist())

    def test_override_threshold_is_used_for_index_split(self):
        vq = _make_nsvq(num_embeddings=10, discarding_threshold=0.5)
        vq.codebooks_used[:] = torch.tensor([20, 2, 2, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)

        _, used_default, _ = vq._get_replacement_indices()
        _, used_override, _ = vq._get_replacement_indices(discarding_threshold=0.05)

        assert set(used_default.tolist()) == {0, 1, 2}
        assert set(used_override.tolist()) == {0, 1, 2, 3}


class TestNSVQReplacement:
    def test_replace_unused_codebooks_resets_usage(self):
        vq = _make_nsvq(num_embeddings=16, discarding_threshold=0.1)
        vq.codebooks_used[0] = 5
        vq.codebooks_used[1] = 1

        vq.replace_unused_codebooks()

        assert int(vq.codebooks_used.sum().item()) == 0

    def test_replace_unused_codebooks_handles_empty_window(self):
        vq = _make_nsvq(num_embeddings=16, discarding_threshold=0.1)
        assert int(vq.codebooks_used.sum().item()) == 0

        vq.replace_unused_codebooks()

        assert int(vq.codebooks_used.sum().item()) == 0

    def test_replace_unused_codebooks_does_not_modify_used_entries(self):
        vq = _make_nsvq(num_embeddings=8, discarding_threshold=0.5)

        # total=6 => expected=0.75 => min_count=0.375
        # used: entries with count >= 0.375 => those with count >= 1
        vq.codebooks_used[:] = torch.tensor([3, 2, 1, 0, 0, 0, 0, 0], dtype=torch.int32)
        unused, used, _ = vq._get_replacement_indices()
        assert set(used.tolist()) == {0, 1, 2}
        assert 3 in set(unused.tolist())

        before = vq.codebooks.detach().clone()
        vq.replace_unused_codebooks()
        after = vq.codebooks.detach().clone()

        # Used entries should be bitwise identical; only unused are overwritten.
        assert torch.equal(after[used], before[used])

    def test_replace_unused_codebooks_honors_override_threshold(self):
        vq = _make_nsvq(num_embeddings=8, discarding_threshold=0.5)
        vq.codebooks_used[:] = torch.tensor([3, 2, 1, 0, 0, 0, 0, 0], dtype=torch.int32)

        # With lower threshold, index 2 should remain used (not replaced).
        unused, used, _ = vq._get_replacement_indices(discarding_threshold=0.1)
        before = vq.codebooks.detach().clone()
        vq.replace_unused_codebooks(discarding_threshold=0.1)
        after = vq.codebooks.detach().clone()

        assert torch.equal(after[used], before[used])
        if len(unused) > 0:
            assert not torch.equal(after[unused], before[unused])
