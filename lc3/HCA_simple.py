import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    return x

# Debug Setting
# class ModelArgs:
#     max_batch_size: int = 4
#     max_seq_len: int = 4096
#     vocab_size: int = 129280
#     dim: int = 4096
#     moe_inter_dim: int = 4096
#     n_layers: int = 7
#     n_hash_layers: int = 0
#     n_mtp_layers: int = 1
#     n_heads: int = 64

#     # mqa
#     q_lora_rank: int = 1024
#     head_dim: int = 512
#     rope_head_dim: int = 64
#     norm_eps: float = 1e-6
#     o_groups: int = 8
#     o_lora_rank: int = 1024
#     window_size: int = 128
#     # compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
#     compress_ratios = [128]



# V4-Pro Setting
class ModelArgs:
    max_batch_size: int = 4
    max_seq_len: int = 4096
    vocab_size: int = 129280
    dim: int = 7168
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 128

    # mqa
    q_lora_rank: int = 1536
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 16
    o_lora_rank: int = 1024
    window_size: int = 128
    # compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_ratios = [128]



def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale=None):
    print('--- [Sparse Attention] ---')
    print('q: ', q.shape)
    print('kv:', kv.shape)
    print('topk_idx:', topk_idxs.shape)
    print('--- [Sparse Attention] ---')
    return q


def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(
            start_pos + 1, window_size),  torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1),
                       (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + \
            torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):

    def __init__(self, args: ModelArgs,
                 compress_ratio: int = 4,
                 head_dim: int = 512):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.compress_ratio = compress_ratio  # 128/4

        # 128 -> 512
        # Learnable positional encoding
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.head_dim))
        self.wkv = nn.Linear(self.dim, self.head_dim)
        self.wgate = nn.Linear(self.dim, self.head_dim)  # What is its purpose?
        self.norm = RMSNorm(self.head_dim, args.norm_eps)

        self.kv_cache = None
        self.kv_state = torch.zeros(args.max_batch_size, compress_ratio, self.head_dim)
        self.score_state = torch.full((args.max_batch_size, compress_ratio, self.head_dim), float("-inf"))

        self.freqs_cis = nn.Parameter(torch.randn(args.max_seq_len))

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        ratio, d, rd = self.compress_ratio, self.head_dim, self.rope_head_dim
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder > 0:
                kv, self.kv_state[:bsz, : remainder] = kv.split(
                    [cutoff, remainder], dim=1)
                self.score_state[:bsz, : remainder] = score[:,
                                                            cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
            self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv = (
                    self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return

        kv = self.norm(kv)
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos +
                                       1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv

    def print(self):
        print('--- Compressor ---')
        print('kv_state: ', self.kv_state.shape)
        print('score_state: ', self.score_state.shape)
        print('kv_cache: ', self.kv_cache.shape)
        print('--- Compressor ---')


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads

        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank

        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(args.n_heads))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank,
                              self.n_heads * self.head_dim)

        self.wkv = nn.Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups,
                              self.n_groups * args.o_lora_rank)

        self.wo_b = nn.Linear(self.n_groups * args.o_lora_rank, self.dim)

        if self.compress_ratio:
            self.compressor = Compressor(args,
                                         self.compress_ratio,
                                         self.head_dim)  


        kv_cache_size = args.window_size + \
            (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.kv_cache = torch.zeros(
            args.max_batch_size, kv_cache_size, self.head_dim)

        self.freqs_cis = nn.Parameter(torch.randn(args.max_seq_len))

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        n_heads = self.n_heads
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        if self.compress_ratio and self.compressor.kv_cache is None:
            # First compress into KV Cache
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis

        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (n_heads,
                                        self.head_dim))  # 65546 -> 128x512
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)  # 448 + 64

        # win kv & topk_idxs
        kv = self.wkv(x)  # dim -> head_dim
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)  # per head 448 + 64
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            compress_topk_idxs = get_compress_topk_idxs(
                ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            # Prefill
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win  # 200 % 128 -> 72
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz,
                                                                :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
                else:
                    print('no compression')

            o = sparse_attn(q, kv, self.attn_sink, topk_idxs)
        else:
            # Decoding
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1) 
            if self.compress_ratio:
                self.compressor(x, start_pos)  
            o = sparse_attn(
                q, self.kv_cache[:bsz], self.attn_sink, topk_idxs)

        apply_rotary_emb(o[..., -rd:],
                         freqs_cis,
                         True)  # why derotate

        # o
        print(f'o.shape: ', o.shape)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        print(f'o.view.shape: ', o.shape)

        print(f'wo_a.shape: ', self.wo_a.weight.shape)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        print(f'wo_a.view.shape: ', wo_a.shape)

        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)  # B,L, [16, 1024]
        print(f'o@wo_a.shape: ', o.shape)

        print(f'wo_b:, {self.wo_b}')
        x = self.wo_b(o.flatten(2))
        print(f'o.flatten(2).shape: ', o.flatten(2).shape)
        print(f'x.shape: ', x.shape)

        print('attention KV-Cache:', self.kv_cache.shape)

        self.compressor.print()

        return x

    def print(self):
        print('--- Attention ---')
        print('kv_cache: ', self.kv_cache.shape)
        print('--- Attention ---')


if __name__ == "__main__":
    args = ModelArgs()
    model = Attention(args=args, layer_id=0)

    seq_len = 511
    # prefill
    print('=' * 30, seq_len)
    X = torch.randn(4, seq_len, args.dim)
    O = model(X, start_pos=0)
    seq_len += 1
    # KV Cache 511

    # decoding
    for i in range(3):
        print('=' * 30, seq_len)
        X = torch.randn(4, 1, args.dim)
        O = model(X, start_pos=seq_len)
        seq_len += 1
