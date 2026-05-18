import torch

torch.manual_seed(42)


def compress_mean_pooling(X, ratio):
    B, L, D = X.shape
    cutoff = (L // ratio) * ratio
    X_blocks = X[:, :cutoff].reshape(B, -1, ratio, D)
    Xc = X_blocks.mean(dim=2)
    return Xc


def compress_weighted(X, ratio, W=None):
    B, L, D = X.shape
    cutoff = (L // ratio) * ratio
    X_blocks = X[:, :cutoff].reshape(B, -1, ratio, D)
    if W is None:
        W = torch.softmax(torch.randn(ratio), dim=0)
    W = W.view(1, 1, -1, 1)
    Xc = (X_blocks * W).sum(dim=2)
    return Xc


# === Demo ===
B, L, D = 2, 16, 8
ratio = 4
X = torch.randn(B, L, D)
Xc_mean = compress_mean_pooling(X, ratio)
Xc_weighted = compress_weighted(X, ratio)
print(f"Input:           {list(X.shape)}  (batch={B}, seq={L}, dim={D})")
print(f"Mean pool out:   {list(Xc_mean.shape)}  ({L} -> {L//ratio} blocks)")
print(f"Weighted out:    {list(Xc_weighted.shape)}")
print()

print("Compression visualization (batch=0, dim=0):")
print(f"  Original [{L}] : {X[0, :, 0].tolist()}")
print(f"  Block1 [{ratio}]: {X[0, :ratio, 0].tolist()}")
print(f"  Block2 [{ratio}]: {X[0, ratio:2*ratio, 0].tolist()}")
print(f"  Mean pool:        [{Xc_mean[0, :, 0].tolist()}]")
print()

print(f"  ratio={ratio}: {L} tokens -> {L//ratio} compressed vectors")
print(f"  V4-Pro ratio=128: 1M tokens -> {1000000//128} vectors")
print()
