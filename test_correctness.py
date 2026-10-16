"""Properties that must hold for every attention type and encoding.

Run: python test_correctness.py   (or: python -m pytest test_correctness.py)

Two of these exist because of defects that raised nothing. Multi-head
attention packed (batch * head) and unpacked (head, batch), permuting outputs
across batch rows for every batch_size > 1. And get_attn_output built
MultiHeadAttention without passing the selected encoding, so all three
multi-head options silently ran the default absolute branch.
"""
import itertools

import torch

from attentions import MultiHeadAttention, get_attn_output, get_candidate_heads
from positional_encoders import (AbsolutePositionalEncoder,
                                 T5RelativePositionalEncoder)

ATTNS = ["scaleddotproduct", "multihead"]
PES = ["abs", "rel", "t5"]


def test_batch_independence():
    """Row i of a batched forward must equal that row run on its own."""
    for pe in PES:
        torch.manual_seed(0)
        mha = MultiHeadAttention(8, 4, pe, dropout_rate=0.0).eval()
        x = torch.randn(4, 6, 8)
        bias = (T5RelativePositionalEncoder(4)(6, 6) if pe == "t5" else None)
        with torch.no_grad():
            batched = mha(x, x, x, relative_bias=bias, is_dropout=False)[0]
            for i in range(x.size(0)):
                solo = mha(x[i:i + 1], x[i:i + 1], x[i:i + 1],
                           relative_bias=bias, is_dropout=False)[0]
                assert torch.allclose(batched[i], solo[0], atol=1e-5), \
                    f"{pe}: row {i} differs when run in a batch"


def test_each_encoding_actually_changes_the_computation():
    """abs, rel and t5 must not collapse onto the same computation."""
    x = torch.randn(2, 5, 8)
    for attn in ATTNS:
        out = {}
        for pe in PES:
            torch.manual_seed(0)
            with torch.no_grad():
                out[pe] = get_attn_output(x, attn, pe, 4)[0]
        for a, b in itertools.combinations(PES, 2):
            assert not torch.allclose(out[a], out[b], atol=1e-7), \
                f"{attn}: {a} and {b} produced identical output"


def test_absolute_encoding_is_only_applied_when_selected():
    """rel and t5 are defined against the raw embedding, not on top of abs."""
    import attentions as A
    fired = {}
    original = AbsolutePositionalEncoder.forward
    AbsolutePositionalEncoder.forward = (
        lambda self, x: (fired.__setitem__("hit", True), original(self, x))[1])
    try:
        x = torch.randn(1, 5, 8)
        for attn, pe in itertools.product(ATTNS, ("rel", "t5")):
            fired.clear()
            A.get_attn_output(x, attn, pe, 4)
            assert not fired.get("hit"), f"{attn}/{pe}: absolute encoding leaked in"
        for attn in ATTNS:
            fired.clear()
            A.get_attn_output(x, attn, "abs", 4)
            assert fired.get("hit"), f"{attn}/abs: absolute encoding was not applied"
    finally:
        AbsolutePositionalEncoder.forward = original


def test_mask_is_applied():
    for pe in PES:
        torch.manual_seed(0)
        mha = MultiHeadAttention(8, 4, pe, dropout_rate=0.0).eval()
        x = torch.randn(2, 5, 8)
        mask = torch.zeros(2, 5, 5, dtype=torch.bool)
        mask[:, :, 3:] = True
        bias = (T5RelativePositionalEncoder(4)(5, 5) if pe == "t5" else None)
        with torch.no_grad():
            _, attn = mha(x, x, x, mask=mask, relative_bias=bias, is_dropout=False)
        assert attn[..., 3:].max() < 1e-6, f"{pe}: masked keys got attention"


def test_t5_bias_head_count_matches_the_model():
    """The bias was built from the requested head count and the model from the
    adjusted one, which disagree whenever the request does not divide emb_dim."""
    x = torch.randn(2, 5, 8)
    for requested in (3, 5, 7):
        get_attn_output(x, "multihead", "t5", requested)   # must not broadcast-fail


def test_candidate_heads_picks_the_nearest_divisor():
    import warnings
    assert get_candidate_heads(768, 8) == 8
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert get_candidate_heads(768, 7) == 8       # 6 and 8 tie; larger wins
        assert get_candidate_heads(100, 8) == 10
        assert len(w) == 2, "a silent head change is the bug this guards"


def test_unknown_options_are_rejected():
    x = torch.randn(1, 5, 8)
    for bad in (("softmax", "abs"), ("multihead", "sinusoid")):
        try:
            get_attn_output(x, bad[0], bad[1], 4)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should raise ValueError, not return None")


def test_modules_move_with_to_and_round_trip_state_dict():
    for pe in PES:
        torch.manual_seed(0)
        a = MultiHeadAttention(8, 4, pe, dropout_rate=0.0).eval()
        torch.manual_seed(1)
        b = MultiHeadAttention(8, 4, pe, dropout_rate=0.0).eval()
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 5, 8)
        bias = (T5RelativePositionalEncoder(4)(5, 5) if pe == "t5" else None)
        with torch.no_grad():
            assert torch.allclose(a(x, x, x, relative_bias=bias, is_dropout=False)[0],
                                  b(x, x, x, relative_bias=bias, is_dropout=False)[0],
                                  atol=1e-6)


def test_absolute_encoding_matches_the_paper():
    import math
    emb_dim, pos = 8, 3
    pe = AbsolutePositionalEncoder(emb_dim)(torch.zeros(1, 5, emb_dim))[0, pos]
    for i in range(emb_dim // 2):
        denom = 10000 ** (2 * i / emb_dim)
        assert math.isclose(pe[2 * i].item(), math.sin(pos / denom), abs_tol=1e-5)
        assert math.isclose(pe[2 * i + 1].item(), math.cos(pos / denom), abs_tol=1e-5)


def test_t5_buckets_are_logarithmic_and_bounded():
    enc = T5RelativePositionalEncoder(num_heads=4, num_buckets=32)
    b = enc.relative_position_bucket(torch.arange(-300, 300)[None, :])
    assert int(b.max()) < 32 and int(b.min()) >= 0
    assert enc.embeddings_table.num_embeddings <= 64, \
        "bucketed bias should need tens of rows, not max_position squared"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            msg = (str(e).splitlines() or [type(e).__name__])[0]
            print(f"  FAIL  {fn.__name__}: {msg[:88]}")
    print(f"\n{len(fns)-len(failed)} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
