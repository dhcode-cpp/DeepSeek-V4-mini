# simplify lc1/kernel.py

import torch
import math


# @torch.jit.script
def sparse_linear_attn_torch(
    q: torch.Tensor,          # (L, H, d)
    kv: torch.Tensor,         # (2, L, d)
    topk_ids: torch.Tensor,   # (L, K)
    block: int = 64,
    scale: float = None,
    attn_sink: torch.Tensor = None,  # skip attn sink
) -> torch.Tensor:
    L, H, d = q.shape
    K_topk = topk_ids.shape[1]
    num_blocks = math.ceil(K_topk / block)
    o = torch.zeros(L, H, d, device=q.device, dtype=q.dtype)

    # Parallel for each query
    for i in range(L):
        qi = q[i]
        idxs_all = topk_ids[i]

        acc_o = torch.zeros(H, d, device=q.device, dtype=q.dtype)

        for t in range(num_blocks):
            start = t * block
            end = min(start + block, K_topk)
            idxs = idxs_all[start:end]

            # gather
            kv_block = kv[:, idxs] # KV 不连续

            scores = qi @ kv_block[0].transpose(0, 1)
            acc_o += scores @ kv_block[1]

        o[i] = acc_o

    return o

if __name__ == "__main__":
    torch.manual_seed(42)
    
    L, H, d = 4, 8, 32
    L_compressed = 6
    
    # QKV
    q = torch.randn(L, H, d)
    kv = torch.randn(2, L + L_compressed, d) # original KV + compress KV
    
    # ids
    num_win_ids = 2
    num_sparse_ids = 3
    num_compressed_ids = 1
    K = num_win_ids + num_sparse_ids + num_compressed_ids
    
    topk_ids = torch.randint(0, L + L_compressed, (L, K)) 

    # sparse attn
    out = sparse_linear_attn_torch(
        q, kv, topk_ids, block=3) 
    
    print(out.shape)
