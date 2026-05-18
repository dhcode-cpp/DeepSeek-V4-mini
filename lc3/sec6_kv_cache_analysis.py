"""
Sec6: KV Cache Size Calculation & Analysis + Overlap Analysis
"""

import torch

torch.manual_seed(42)

def kv_cache_size(window_size, max_seq_len, ratio):
    win_cache = window_size
    compressed_cache = max_seq_len // ratio if ratio else 0
    total = win_cache + compressed_cache
    return win_cache, compressed_cache, total

win_cache, comp_cache, total = kv_cache_size(128, 4096, 128)
print(win_cache, comp_cache, total)
print()

# === V4-Pro Configuration ===
win_cache_p, comp_cache_p, total_p = kv_cache_size(128, 1000000, 128)
print("V4-Pro (1M context, ratio=128):")
print(f"  Window Cache:       {win_cache_p:>6d} KV slots")
print(f"  Compressed Cache:   {comp_cache_p:>6d} KV slots (1M/128)")
print(f"  Total:              {total_p:>6d} KV slots")
print(f"  Equivalent uncompressed KV:      {1000000:>6d} slots")
print(f"  Compression ratio:             {1000000 / total_p:.1f}x")
print()

print("Compared to standard Full Attention:")
print(f"  Standard KV Cache (1M): {1000000:>6d} slots")
print(f"  HCA KV Cache:       {total_p:>6d} slots")
print(f"  Reduction:               {1000000 - total_p:>6d} slots ({((1 - total_p/1000000)*100):.1f}%)")
print()


# ============================================================
# Window + Compressed Overlap Analysis
# ============================================================

print("=" * 70)
print("Window ID vs Compressed ID Overlap")
print("=" * 70)

win_size = 3
ratio = 3
seq = 7
print(f"Sequence length={seq}, window={win_size}, ratio={ratio}")
print()

    # Compute window ids for each token
win_ids = [list(range(max(0, i-win_size+1), i+1)) for i in range(seq)]
    # First token position of compressed block (block index * ratio)
comp_blocks = list(range(0, seq, ratio))

print("Window indices (recent KV visible to each token):")
for i, ids in enumerate(win_ids):
    print(f"  token {i}: {ids}")
print()
print(f"Compressed block first token positions: {comp_blocks}")
print(f"  Note: Block 0=[0,1,2] -> compressed id=0")
print(f"        Block 1=[3,4,5] -> compressed id=1")
print(f"        Block 2=[6]     -> compressed id=2 (incomplete)")
print()

print("Overlap observation:")
print("  token 3 window ids: [1,2,3]")
print(f"  Token 1 (block 0) overlaps with compressed id 0")
print(f"  Window KV and compressed KV overlap at token level")
print(f"  But attention does not distinguish sources; concatenated then softmaxed")
print()
