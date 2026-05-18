import torch
from torch import nn
torch.manual_seed(42)

class GatedCompressor(nn.Module):
    def __init__(self, dim, head_dim, ratio):
        super().__init__()
        self.ratio = ratio
        self.wkv = nn.Linear(dim, head_dim, bias=False)
        self.wgate = nn.Linear(dim, head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(ratio, head_dim))
        nn.init.normal_(self.ape, std=0.02)

    def forward(self, x):
        B, L, D = x.shape
        ratio = self.ratio
        n_blocks = L // ratio
        cutoff = n_blocks * ratio

        kv = self.wkv(x)
        score = self.wgate(x)
        print(f"  wkv(x):   {list(kv.shape)}")
        print(f"  wgate(x): {list(score.shape)}")

        kv = kv[:, :cutoff].reshape(B, n_blocks, ratio, -1)
        score = score[:, :cutoff].reshape(B, n_blocks, ratio, -1)
        print(f"  kv blocks:     {list(kv.shape)}")
        print(f"  score blocks:  {list(score.shape)}")

        score = score + self.ape
        weight = score.softmax(dim=2)
        print(f"  weight (softmax): {list(weight.shape)}")

        kv_c = (kv * weight).sum(dim=2)
        return kv_c, weight


# === Demo ===
dim, head_dim = 32, 16
ratio = 4
B, L = 2, 16
x = torch.randn(B, L, dim)
comp = GatedCompressor(dim, head_dim, ratio)

print(f"\nGatedCompressor input/output:")
kv_c, w = comp(x)
print(f"  compressed KV: {list(kv_c.shape)}  ({L} -> {L//ratio})")
print()

print("Gate weight example (batch=0, block=0):")
w_block0 = w[0, 0]
print(f"  shape: {list(w_block0.shape)}")
print(f"  weights[:, :4]:")
print(w_block0[:, :4])
print(f"  sum over ratio dim (should be 1.0):")
print(w_block0.sum(dim=0)[:4].tolist())
print()
