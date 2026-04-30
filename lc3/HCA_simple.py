
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
    def forward(self, x: torch.Tensor):
        return x


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    return x

class ModelArgs:
    max_batch_size: int = 4
    max_seq_len: int = 4096
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64

    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    # compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_ratios = [128]


def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class CompressorHCA(nn.Module):

    def __init__(self, args: ModelArgs,
                 compress_ratio: int = 4,
                 head_dim: int = 512):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim  
        self.rope_head_dim = args.rope_head_dim 
        self.compress_ratio = compress_ratio  

        self.ape = nn.Parameter(torch.empty(compress_ratio, self.head_dim))
        self.wkv = nn.Linear(self.dim, self.head_dim)
        self.wgate = nn.Linear(self.dim, self.head_dim)  # 有什么作用？
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None
        self.register_buffer("kv_state", torch.zeros(
            args.max_batch_size, compress_ratio, self.head_dim, ), persistent=False)
        self.register_buffer("score_state", torch.full(
            (args.max_batch_size, compress_ratio, self.head_dim), float("-inf")), persistent=False)
        self.freqs_cis: torch.Tensor = None

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, d, rd = self.compress_ratio, self.head_dim, self.rope_head_dim
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio  # 200 >= 128
            remainder = seqlen % ratio  # 72
            cutoff = seqlen - remainder  # 200 - 72 = 128
            if remainder > 0:
                # compressor 层面 kv cache起来
                # CSA/HCA 都一样
                # KVCache 是驱逐的
                # Compressor 后的 cache起来吗？还是他的cache 在外层？
                kv, self.kv_state[:bsz, : remainder] = kv.split(
                    [cutoff, remainder], dim=1)
                self.score_state[:bsz, : remainder] = score[:,
                    cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))  # bsz, n, 128, d
            score = score.unflatten(1, (-1, ratio)) + self.ape  # score 负责处理位置。
            kv = (kv * score.softmax(dim=2)).sum(dim=2)  # bsz, n, 1, d
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


class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression.
    Uses low-rank Q projection (wq_a -> q_norm -> wq_b) and grouped low-rank O projection."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim  # 7168?
        self.n_heads = args.n_heads  # 128

        self.q_lora_rank = args.q_lora_rank  # 1536
        self.o_lora_rank = args.o_lora_rank  # 1024

        self.head_dim = args.head_dim  # 512
        self.rope_head_dim = args.rope_head_dim  # 64
        self.n_groups = args.o_groups  # 16
        self.window_size = args.window_size  # 128
        self.compress_ratio = args.compress_ratios[layer_id]  # 128
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(args.n_heads)
        self.wq_a=nn.Linear(self.dim, self.q_lora_rank)  # dim -> 1536
        self.q_norm=RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b=nn.Linear(self.q_lora_rank, 
                            self.n_heads * self.head_dim)  # 128x512 -> 65536

        self.wkv=nn.Linear(self.dim, self.head_dim)  # dim -> 512
        self.kv_norm=RMSNorm(self.head_dim, self.eps)

        # 65536//16 = 4096, 16 x 1024 = 16384
        self.wo_a=nn.Linear(self.n_heads * self.head_dim // self.n_groups,
                               self.n_groups * args.o_lora_rank)

        # 16384, dim
        self.wo_b=nn.Linear(self.n_groups * args.o_lora_rank, self.dim)

        # self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor=Compressor(args,
                                         self.compress_ratio,
                                         self.head_dim)  # 512

        # winsize + (c_kv - 1)
        kv_cache_size=args.window_size +
            (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size,
                                                     kv_cache_size,
                                                     self.head_dim), persistent=False)


    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _=x.size()
        freqs_cis=self.freqs_cis[start_pos:start_pos+seqlen]
        win=self.window_size
        ratio=self.compress_ratio
        rd=self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache=self.kv_cache[:, win:]
            self.compressor.freqs_cis=self.freqs_cis


        # q
        qr=q=self.q_norm(self.wq_a(x))
        q=self.wq_b(q).unflatten(-1, (self.n_local_heads,
                    self.head_dim))  # 65546 -> 128x512
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)  # 每头 448 + 64

        # win kv & topk_idxs
        kv=self.wkv(x)  # dim -> head_dim
        kv=self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)  # 每头 448 + 64
        topk_idxs=get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset=kv.size(1) if start_pos == 0 else win

            # 这个分数怎么来的
            # 所以这里应该是 Pre-topk 压缩
            # 输出是什么？

            compress_topk_idxs=get_compress_topk_idxs(
                ratio, bsz, seqlen, start_pos, offset)


            # 块 ID 还是 key ID？
            topk_idxs=torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs=topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            # Prefill
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen]=kv
            else:
                cutoff=seqlen % win  # 200 % 128 -> 72
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz,
                    :cutoff]=kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                # 这个符号是什么？
                # KV 是prefill的(swa)
                # 压缩时候，不是压缩KV!!
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    # 大家的 KV 都是 head dim？
                    kv=torch.cat([kv, kv_compress], dim=1)

            # 算子实现细节
            o=sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            # Decoding
            self.kv_cache[:bsz, start_pos % win]=kv.squeeze(1)  # KV 在一个固定window内
            if self.compress_ratio:
                self.compressor(x, start_pos)  # 单个 token 如何压缩内？
            o=sparse_attn(
                q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)

        apply_rotary_emb(o[..., -rd:],
                         freqs_cis,
                         True)  # 这里有意思了，为什么输出o也要用位置编码？， 为什么要derotate

        # o
        o=o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a=self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # NOTE: wo_a is FP8 in checkpoint; could do FP8 einsum here for better perf,
        # but using BF16 for simplicity.
        o=torch.einsum("bsgd,grd->bsgr", o, wo_a)  # B,L, [16, 1024]
        x=self.wo_b(o.flatten(2))
        return x