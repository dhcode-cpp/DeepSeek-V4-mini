import torch
from torch import nn
from hca_common import precompute_freqs_cis, apply_rotary_emb, RMSNorm

torch.manual_seed(42)

class Compressor(nn.Module):

    def __init__(self, dim, head_dim, rope_head_dim, ratio, max_batch_size=2):
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
        ratio = self.ratio
        rd = self.rope_head_dim
        dtype = x.dtype

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if debug:
            print(f"    wkv(x):   {list(kv.shape)}")
            print(f"    wgate(x): {list(score.shape)}")

        if start_pos == 0:
            # ===== Prefill =====
            should_compress = L >= ratio
            remainder = L % ratio
            cutoff = L - remainder
            if remainder > 0:
                kv, self.kv_state[:B, :remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:B, :remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
                if debug:
                    print(f"    remainder={remainder}, cached in kv_state")
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            if debug:
                print(f"    compressed KV blocks: {list(kv.shape)}")
        else:
            # ===== Decode =====
            should_compress = (start_pos + 1) % ratio == 0
            score += self.ape[start_pos % ratio]
            self.kv_state[:B, start_pos % ratio] = kv.squeeze(1)
            self.score_state[:B, start_pos % ratio] = score.squeeze(1)
            if debug:
                fill = start_pos % ratio
                print(f"    pos_in_block={fill+1}/{ratio}, kv_state[{fill}] = kv_t")
            if should_compress:
                s = self.score_state[:B].softmax(dim=1)
                kv = (self.kv_state[:B] * s).sum(dim=1, keepdim=True)
                if debug:
                    print(f"    [TRIGGER] compressed: {list(kv.shape)}")

        if not should_compress and start_pos != 0:
            return None

        kv = self.norm(kv.to(dtype))
        # Inter-block positional encoding YaRN (applied to last rope_head_dim dims)
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


if __name__ == "__main__":
    # === Demo: Prefill ===
    print("\n--- Compressor Demo: Prefill ---")
    B, dim, head_dim, rd = 2, 32, 16, 4
    ratio = 4

    compressor = Compressor(dim, head_dim, rd, ratio)
    freqs_cis = precompute_freqs_cis(rd, 32, base=40000.0)
    kv_cache = torch.zeros(B, 8, head_dim)
    compressor.set_kv_cache(kv_cache, freqs_cis)

    x = torch.randn(B, 10, dim)
    print(f"Prefill: seqlen={x.size(1)}, ratio={ratio}")
    out = compressor(x, start_pos=0, debug=True)
    print(f"  output: {'None' if out is None else list(out.shape)}")
    print()

    # === Demo: Decode ===
    print("--- Compressor Demo: Decode (r=4, step by step) ---")
    compressor2 = Compressor(dim, head_dim, rd, ratio)
    compressor2.set_kv_cache(torch.zeros(B, 8, head_dim), freqs_cis)

    for pos in range(8):
        x_t = torch.randn(B, 1, dim)
        out = compressor2(x_t, start_pos=10 + pos, debug=True)
        status = "COMPRESSED!" if out is not None else "cached"
        print(f"  pos={10+pos}: {status}")
        print()

    print("=" * 70)
    print("Sec4.1: Intra-block Position Encoding APE (Intra-block)")
    print("=" * 70)

    print(f"APE shape: {list(compressor.ape.shape)} = [{ratio}, {head_dim}]")
    print(f"i.e., ratio={ratio} positions, each with head_dim={head_dim}-dim encoding")
    print()
    print("APE function: score = wgate(x) + APE (before softmax)")
    print("Model learns compression weight preferences for different positions within a block via APE")
    print()

    ape_data = compressor.ape.detach()
    print(f"APE values (first 4 dims):")
    for i in range(ratio):
        print(f"  pos {i}: {ape_data[i, :4].tolist()}")
    print()

    def yarn_position_ids(block_idx, ratio):
        first_pos = block_idx * ratio
        last_pos = block_idx * ratio + ratio - 1
        print(f"    Block {block_idx}: tokens=[{first_pos}, {last_pos}]")
        print(f"      pos_id (first token):   {first_pos}")
        print(f"      yarn_factor (last token): {last_pos}")
        return first_pos

    print("YaRN encoding rules:")
    print("  1. RoPE position ID = first token position of the block")
    print("  2. YaRN scaling factor = last token position of the block")
    print("  3. Encoding applied to last rope_head_dim dims (V4 Pro: 64/512)")
    print()

    print("Inter-block positional encoding example (ratio=4):")
    for bi in range(4):
        yarn_position_ids(bi, ratio=4)
    print()

    print("Key implementation code:")
    print("  freqs_cis = self.freqs_cis[:cutoff:ratio]     # Prefill")
    print("  freqs_cis = self.freqs_cis[start_pos+1-ratio]  # Decode")
    print()
