import torch
from torch import nn


def precompute_freqs_cis(dim, seqlen, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis, inverse=False):
    if freqs_cis is None:
        return x
    y = x
    rd = freqs_cis.size(-1) * 2
    x_rope = x[..., -rd:].float()
    x_complex = torch.view_as_complex(x_rope.unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    ndim = x_complex.ndim
    if ndim == 3:
        freqs_view = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    elif ndim == 4:
        freqs_view = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    else:
        freqs_view = freqs_cis.view(1, 1, x_complex.size(-1))
    x_rotated = torch.view_as_real(x_complex * freqs_view).flatten(-2)
    y[..., -rd:] = x_rotated.to(x.dtype)
    return y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        var = x.float().square().mean(-1, keepdim=True)
        return (self.weight * x.float() * torch.rsqrt(var + self.eps)).to(x.dtype)
