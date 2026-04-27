# simplest-version DeepSeek-V4
# 1. show model components and dataflow
# 2. skip compute details
# 3. easy debug every block input-output tensor shape

from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn

torch.manual_seed(0)


@dataclass
class ModelArgs:
    # vocab_size: int = 129280
    vocab_size: int = 100
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 8

    # MoE
    # layer(1-2) HASH
    # layer( >2) Top
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_hash_layers: int = 2 # moe type control

    # Attn Type
    #   0: MQA
    #   4: CSA
    # 128: HCA
    compress_ratios = [0, 0, 4, 128, 4, 128, 4, 0]  # for switch MQA/CSA/HCA
    head_dim: int = 512

    # hc
    hc_mult: int = 4


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.compress_ratio = args.compress_ratios[layer_id]

        if self.compress_ratio:
            if self.compress_ratio == 4:  # 4 is CSA
                self.layer_cls = "CSA"
            else:
                self.layer_cls = "HCA"
        else:
            self.layer_cls = "MQA"

    def forward(self, x: torch.Tensor, start_pos: int):
        print(f' > block.{self.layer_id} attn type: {self.layer_cls}')
        return x


class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.layer_id = layer_id
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Linear(self.dim, args.n_routed_experts)
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size,
                                                    args.n_activated_experts,
                                                    dtype=torch.int32),
                                        requires_grad=False)
            self.moe_type = 'HASH'
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts,
                                                 dtype=torch.float32))
            self.moe_type = 'TOPK'

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        print(f' > block.{self.layer_id} moe type: {self.moe_type}')
        return None, None


class Expert(nn.Module):
    def __init__(self, dim: int, moe_inter_dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class MoE(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_activated_experts = args.n_activated_experts
        self.gate = Gate(layer_id, args)  # hash id in gate manage
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim)
                                      for _ in range(args.n_routed_experts)])
        self.shared_experts = Expert(args.dim, args.moe_inter_dim)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        self.gate(x)  # skip moe compute
        return x


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        self.hc_mult = args.hc_mult

    def hc_pre(self, x: torch.Tensor):
        y = torch.sum(x, dim=2)
        return y

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor):
        y = x.unsqueeze(-2) + residual
        return y

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x = self.hc_pre(x)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual)

        residual = x
        x = self.hc_pre(x)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)  # for MoE
        x = self.hc_post(x, residual)
        return x


class LMHeadWithHC(nn.Module):

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor, norm: RMSNorm):
        x = self.hc_head(x, norm) # B, L, HC, D -> B, L, D
        logits = self.weight(x)
        return logits

    def hc_head(self, x: torch.Tensor, norm):
        # fuse kernel 
        # hc head_merge with weight and norm
        y = torch.sum(x, dim=2)
        return y


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_eps = 1e-6
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))

        self.norm = RMSNorm(args.dim, self.norm_eps)  # norm merge to lm-head
        # self.head = nn.Linear(args.vocab_size, args.dim)
        self.head = LMHeadWithHC(args.vocab_size, args.dim)

        # skip mtp
        # self.mtp = torch.nn.ModuleList()

        self.hc_mult = args.hc_mult

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        h = self.embed(input_ids)

        # HC expand
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
            print('-'*50)

        # HC merge + rms_norm + lm_head proj
        logits = self.head(h, self.norm)
        return logits


if __name__ == "__main__":
    args = ModelArgs(n_hash_layers=2)
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)

    # forward
    logits = model(x)
    print('input shape:', x.shape)
    print('output logits shape:', logits.shape)

    # in original generate demo
    # 1. decoding token_ids and with logits
    # 2. forward to every mtp_head by input (logits, input_ids)