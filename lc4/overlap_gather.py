# ./lc4/overlap_gather.py
# Compute-Communication Overlap for Sparse Attention

import torch
import math

torch.manual_seed(42)


def overlap_gather_attn(q, kv, topk_ids, block=64):
    """Sparse attention with compute-communication overlap.

    For each query, topk_ids are split into blocks. While computing
    attention on block t, the next block t+1 is prefetched asynchronously.

    Args:
        q:           [L, H, d]          query
        kv:          [L_kv, d]          KV (MQA: K and V share same tensor)
        topk_ids:    [L, K]             per-query candidate indices
        block:        int                KV gather block size
    Returns:
        o:           [L, H, d]          output
    """
    L, H, d = q.shape
    K_topk = topk_ids.shape[1]
    num_blocks = math.ceil(K_topk / block)
    o = torch.zeros(L, H, d)

    for i in range(L):
        qi = q[i]
        idxs_all = topk_ids[i]

        # Simulate async prefetch: next_kv holds the KV for the next iteration
        next_kv = None
        if num_blocks > 0:
            start = 0
            end = min(block, K_topk)
            idxs = idxs_all[start:end]
            next_kv = kv[idxs]  # "fetch" first block

        acc_o = torch.zeros(H, d)

        for t in range(num_blocks):
            # 1. Use previously fetched KV block (overlap: next block was
            #    fetched during the previous attention computation)
            kv_block = next_kv

            # 2. Async prefetch the NEXT block while computing current block
            if t + 1 < num_blocks:
                next_start = (t + 1) * block
                next_end = min(next_start + block, K_topk)
                next_idxs = idxs_all[next_start:next_end]
                next_kv = kv[next_idxs]  # prefetch t+1 (overlaps with compute)
            else:
                next_kv = None

            # 3. Compute attention on current block (MQA: K=V=kv_block)
            if kv_block is not None:
                scores = qi @ kv_block.T       # [H, blk]
                acc_o += scores @ kv_block      # [H, d]

        o[i] = acc_o

    return o


if __name__ == "__main__":
    L, H, d = 8, 4, 32
    L_kv = 16
    K_topk = 6
    block = 3

    q = torch.randn(L, H, d)
    kv = torch.randn(L_kv, d)
    topk_ids = torch.randint(0, L_kv, (L, K_topk))

    o = overlap_gather_attn(q, kv, topk_ids, block=block)
    print(f"Overlap gather attention output shape: {list(o.shape)}")