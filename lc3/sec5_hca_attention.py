import torch
import torch.nn.functional as F
from torch import nn
from hca_common import precompute_freqs_cis, apply_rotary_emb, RMSNorm

torch.manual_seed(42)


def single_query_sparse_attn(q, K, V):
    S = q @ K.t()
    P = F.softmax(S, dim=-1)
    Z = P @ V
    return Z


def get_window_topk_idxs(window_size, seqlen, start_pos):
    if start_pos > 0:
        sp = start_pos % window_size
        idxs = torch.cat([torch.arange(sp + 1, window_size),
                          torch.arange(0, sp + 1)], dim=0)
        return idxs.unsqueeze(0).expand(seqlen, -1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
        return matrix


def get_compress_topk_idxs(ratio, seqlen, start_pos, offset, max_n=None):
    if start_pos > 0:
        n = (start_pos + 1) // ratio
        if max_n is not None:
            n = min(n, max_n)
        idxs = torch.arange(n) + offset
        return idxs.unsqueeze(0)
    else:
        n = seqlen // ratio
        if max_n is not None:
            n = min(n, max_n)
        matrix = torch.arange(n).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
        return matrix


class Compressor(nn.Module):
    def __init__(self, dim, head_dim, rope_head_dim, ratio, max_batch_size=1):
        super().__init__()
        self.ratio = ratio
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.wkv = nn.Linear(dim, head_dim, bias=False)
        self.wgate = nn.Linear(dim, head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(ratio, head_dim))
        self.norm = RMSNorm(head_dim)
        nn.init.normal_(self.ape, std=0.02)
        self.register_buffer("kv_state", torch.zeros(max_batch_size, ratio, head_dim), persistent=False)
        self.register_buffer("score_state",
                             torch.full((max_batch_size, ratio, head_dim), float("-inf")), persistent=False)
        self.kv_cache = None
        self.freqs_cis = None

    def forward(self, x, start_pos, debug=False):
        B, L, D = x.shape
        ratio, rd = self.ratio, self.rope_head_dim
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = L >= ratio
            remainder = L % ratio
            cutoff = L - remainder
            if remainder > 0:
                kv, self.kv_state[:B, :remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:B, :remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            score += self.ape[start_pos % ratio]
            self.kv_state[:B, start_pos % ratio] = kv.squeeze(1)
            self.score_state[:B, start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:B] * self.score_state[:B].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress and start_pos != 0:
            return None
        kv = self.norm(kv.to(x.dtype))
        if start_pos == 0:
            freqs = self.freqs_cis[:cutoff:ratio]
        else:
            freqs = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs)
        if start_pos == 0:
            self.kv_cache[:B, :L // ratio] = kv
        else:
            self.kv_cache[:B, start_pos // ratio] = kv.squeeze(1)
        return kv

    def set_kv_cache(self, kv_cache, freqs_cis):
        self.kv_cache = kv_cache
        self.freqs_cis = freqs_cis


class HCA(nn.Module):
    def __init__(self, dim, n_heads, head_dim, rope_head_dim, window_size, ratio):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.window_size = window_size
        self.ratio = ratio
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wkv = nn.Linear(dim, head_dim, bias=False)
        self.kv_norm = RMSNorm(head_dim)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.compressor = Compressor(dim, head_dim, rope_head_dim, ratio, max_batch_size=1)

    def set_cache(self, max_seq_len, max_batch_size, freqs_cis):
        cache_size = self.window_size + max_seq_len // self.ratio
        self.kv_cache = torch.zeros(max_batch_size, cache_size, self.head_dim)
        self.compressor.set_kv_cache(self.kv_cache[:, self.window_size:], freqs_cis)
        self.freqs_cis = freqs_cis
        print(f"  kv_cache shape: {list(self.kv_cache.shape)}")

    def forward(self, x, start_pos, debug=False):
        B, L, D = x.shape
        win, ratio, rd = self.window_size, self.ratio, self.rope_head_dim
        freqs_cis = self.freqs_cis[start_pos:start_pos+L]

        if debug:
            print(f"  HCA forward: B={B}, L={L}, start_pos={start_pos}")

        q = self.wq(x).unflatten(-1, (self.n_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        if debug:
            print(f"  Q: {list(q.shape)}")

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if debug:
            print(f"  KV: {list(kv.shape)}")

        topk_ids = get_window_topk_idxs(win, L, start_pos)
        offset = kv.size(1) if start_pos == 0 else win
        compress_ids = get_compress_topk_idxs(ratio, L, start_pos, offset)
        topk_ids = torch.cat([topk_ids, compress_ids], dim=-1).int()
        if debug:
            print(f"  Window IDs: {list(get_window_topk_idxs(win, L, start_pos).shape)}, " +
                  f"Compress IDs: {list(compress_ids.shape)}, Total: {list(topk_ids.shape)}")

        if start_pos == 0:
            if L <= win:
                self.kv_cache[:B, :L] = kv
            else:
                cutoff = L % win
                self.kv_cache[:B, cutoff:win], self.kv_cache[:B, :cutoff] = \
                    kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if ratio:
                kv_c = self.compressor(x, start_pos)
                if kv_c is not None:
                    kv_all = torch.cat([kv, kv_c], dim=1)
        else:
            self.kv_cache[:B, start_pos % win] = kv.squeeze(1)
            if ratio:
                self.compressor(x, start_pos)
            kv_all = self.kv_cache[:B]

        if debug:
            print(f"  KV for attn: {list(kv_all.shape)}, topk_ids max: {topk_ids.max().item()}")

        o = torch.zeros(B, L, self.n_heads, self.head_dim)
        for i in range(L):
            ids = topk_ids[i] if topk_ids.dim() == 2 else topk_ids[0, i]
            valid = ids >= 0
            if valid.any():
                kv_sel = kv_all[0, ids[valid]]
                o[0, i] = single_query_sparse_attn(q[0, i], kv_sel, kv_sel)
        if debug:
            print(f"  Sparse Attn out: {list(o.shape)}")

        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
        x_out = self.wo(o.view(B, L, -1))
        if debug:
            print(f"  Output: {list(x_out.shape)}")
        return x_out


# === Demo ===
if __name__ == "__main__":
    print("\n--- HCA Demo: Prefill ---")
    dim, n_heads, head_dim, rd = 32, 4, 16, 4
    win_size, ratio = 8, 4

    model = HCA(dim, n_heads, head_dim, rd, win_size, ratio)
    freqs_cis = precompute_freqs_cis(rd, 32)
    model.set_cache(32, 1, freqs_cis)

    x = torch.randn(1, 16, dim)
    out = model(x, start_pos=0, debug=True)
    print(f"\n  Final output: {list(out.shape)}")
    print()

    print("--- HCA Demo: Decode (2 steps) ---")
    for step in range(2):
        pos = 16 + step
        x_t = torch.randn(1, 1, dim)
        out_t = model(x_t, start_pos=pos, debug=True)
        print()
