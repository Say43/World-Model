"""Shape and basic-correctness tests for src/model/blocks.py primitives."""
import torch

from src.model.blocks import (
    AdaLNSingle,
    Attention,
    RMSNorm,
    SwiGLU,
    hidden_dim_swiglu,
    modulate,
    sinusoidal_embedding,
)


def test_rmsnorm_shape_and_unit_scale():
    x = torch.randn(2, 5, 32) * 10.0
    norm = RMSNorm(32)
    out = norm(x)
    assert out.shape == x.shape
    # RMS of the normalized output should be close to 1 along the last dim.
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_affine_weight_used():
    x = torch.randn(2, 8)
    norm = RMSNorm(8, elementwise_affine=True)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    out = norm(x)
    out_default = RMSNorm(8, elementwise_affine=False)(x)
    assert torch.allclose(out, out_default * 2.0, atol=1e-5)


def test_swiglu_shape():
    hidden = hidden_dim_swiglu(64, mult=4.0)
    mlp = SwiGLU(64, hidden)
    x = torch.randn(3, 7, 64)
    out = mlp(x)
    assert out.shape == x.shape


def test_hidden_dim_swiglu_multiple_of():
    for dim in (32, 100, 256):
        h = hidden_dim_swiglu(dim, mult=4.0, multiple_of=32)
        assert h % 32 == 0
        assert h > 0


def test_attention_shape_no_mask():
    attn = Attention(dim=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_causal_mask_blocks_future():
    """With a strict lower-triangular mask, changing token j > i must not
    change the output at position i (single-layer sanity check; the full
    model-level causality test lives in test_model_causality.py)."""
    torch.manual_seed(0)
    attn = Attention(dim=16, num_heads=2)
    attn.eval()
    l = 6
    mask = torch.tril(torch.ones(l, l, dtype=torch.bool))
    x = torch.randn(1, l, 16)
    with torch.no_grad():
        out1 = attn(x, attn_mask=mask)
        x2 = x.clone()
        x2[:, 4:] = torch.randn(1, l - 4, 16)  # perturb only late tokens
        out2 = attn(x2, attn_mask=mask)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-6)


def test_adaln_single_zero_init_identity():
    mod = AdaLNSingle(embed_dim=16, dim=8, num_chunks=6)
    t_embed = torch.randn(2, 3, 16)
    out = mod(t_embed)
    assert out.shape == (2, 3, 48)
    assert torch.allclose(out, torch.zeros_like(out))


def test_modulate_identity_at_zero():
    x = torch.randn(2, 4, 8)
    shift = torch.zeros(2, 1, 8)
    scale = torch.zeros(2, 1, 8)
    out = modulate(x, shift, scale)
    assert torch.allclose(out, x)


def test_sinusoidal_embedding_shape():
    t = torch.rand(2, 5)
    emb = sinusoidal_embedding(t, dim=32)
    assert emb.shape == (2, 5, 32)
    emb_odd = sinusoidal_embedding(t, dim=31)
    assert emb_odd.shape == (2, 5, 31)
