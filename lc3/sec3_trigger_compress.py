import torch

torch.manual_seed(42)


def prefill_compression_info(seqlen, ratio):
    n_blocks = seqlen // ratio
    remainder = seqlen % ratio
    cutoff = n_blocks * ratio
    should = seqlen >= ratio
    print(f"  seqlen={seqlen}, ratio={ratio}")
    print(f"    full blocks: {n_blocks} x {ratio} = {cutoff}")
    print(f"    remainder:   {remainder} (cached, not compressed)")
    if should:
        print(f"    compressed:  {n_blocks} vectors")
    else:
        print(f"    [skip: seqlen < ratio]")
    print(f"    compressed_seq / cache_seq = {n_blocks} / {remainder}")
    return should, n_blocks, cutoff, remainder


def decode_compression_info(start_pos, ratio):
    pos_in_block = start_pos % ratio
    should = (start_pos + 1) % ratio == 0
    print(f"  start_pos={start_pos}, ratio={ratio}")
    print(f"    pos_in_block={pos_in_block}/{ratio}")
    print(f"    cache_fill: {pos_in_block + 1}/{ratio}")
    if should:
        print(f"    [TRIGGER] Compress! Generate 1 compressed vector")
    else:
        print(f"    [WAIT] Continue caching...")
    return should, pos_in_block


# === Demo: Prefill ===
print("\n--- Prefill Decision ---")
for L in [3, 4, 7, 8, 15, 16]:
    prefill_compression_info(L, 4)
    print()

# === Demo: Decode ===
print("--- Decode Decision ---")
for pos in range(9):
    decode_compression_info(pos, 4)
    print()



def simulate_prefill_kv_state(seqlen, ratio, head_dim=4):
    print(f"\nPrefill seqlen={seqlen}, ratio={ratio}:")
    kv_state = torch.zeros(ratio, head_dim)
    score_state = torch.full((ratio, head_dim), float("-inf"))

    n_blocks = seqlen // ratio
    remainder = seqlen % ratio
    cutoff = n_blocks * ratio

    if remainder > 0:
        kv_cache_val = torch.randn(remainder, head_dim) * 0.5
        kv_state[:remainder] = kv_cache_val
        score_state[:remainder] = torch.randn(remainder, head_dim) + 1.0

    print(f"  Full blocks: {n_blocks}, compressed to {n_blocks} vectors")
    print(f"  Remainder: {remainder} tokens left in cache")
    if remainder > 0:
        print(f"  kv_state[:{remainder}] filled, score_state[:{remainder}] filled")
        print(f"\n  Cache state (first 2 dims):")
        print(f"  kv_state[:{remainder}]:")
        print(kv_state[:remainder, :2].tolist())
        print(f"  score_state[:{remainder}]:")
        print(score_state[:remainder, :2].tolist())


simulate_prefill_kv_state(6, 4)
simulate_prefill_kv_state(8, 4)
print()
