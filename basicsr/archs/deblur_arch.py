import time
from basicsr.utils.registry import ARCH_REGISTRY
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

NEG_INF = -1000000
class ChannelAttention(nn.Module):
    def __init__(self, num_feat):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat, 1, padding=0,bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.attention(x)
        y = self.sig(y)
        return x * y,y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            # nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            # nn.GELU(),
            # nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat)
        )

    def forward(self, x):
        out,y=self.cab(x)
        return out,y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SS2D(nn.Module):#This is the code of R-SSM
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=False,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),         
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj
        #add
        self.x_proj2 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight2 = nn.Parameter(torch.stack([t.weight for t in self.x_proj2], dim=0))  # (K=4, N, inner)
        del self.x_proj2

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        #add
        self.dt_projs2 = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight2 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs2], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias2 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs2], dim=0))  # (K=4, inner)
        del self.dt_projs2

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=4, D, N)
        #add
        self.A_logs2 = self.A_log2_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=4, D, N)
        self.Ds2 = self.D2_init(self.d_inner, copies=2, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.selective_scan2 = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @staticmethod
    def A_log2_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D2_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core3(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 2#4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],dim=1).view(B, 2, -1, L)
        xs=x_hwwh
        #xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L).contiguous()
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L).contiguous()
        Cs = Cs.float().view(B, K, -1, L).contiguous()  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1).contiguous()
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state).contiguous()
        dt_projs_bias = self.dt_projs_bias.float().view(-1).contiguous()  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L).contiguous()
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, 1, -1, L).contiguous()
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        #invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0],inv_y[:,0]
        #return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 2
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs=x_hwwh
        #xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight2)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight2)
        xs = xs.float().view(B, -1, L).contiguous()
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L).contiguous()
        Cs = Cs.float().view(B, K, -1, L).contiguous()  # (b, k, d_state, l)
        Ds = self.Ds2.float().view(-1).contiguous()
        As = -torch.exp(self.A_logs2.float()).view(-1, self.d_state).contiguous()
        dt_projs_bias = self.dt_projs_bias2.float().view(-1).contiguous()  # (k * d)
        out_y = self.selective_scan2(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L).contiguous()
        assert out_y.dtype == torch.float
        #inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L).contiguous()
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        #invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], wh_y

    def forward_core2(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 2
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs=x_hwwh
        #xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight2)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight2)
        xs = xs.float().view(B, -1, L).contiguous()
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L).contiguous()
        Cs = Cs.float().view(B, K, -1, L).contiguous()  # (b, k, d_state, l)
        Ds = self.Ds2.float().view(-1).contiguous()
        As = -torch.exp(self.A_logs2.float()).view(-1, self.d_state).contiguous()
        dt_projs_bias = self.dt_projs_bias2.float().view(-1).contiguous()  # (k * d)
        out_y = self.selective_scan2(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L).contiguous()
        assert out_y.dtype == torch.float
        #inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L).contiguous()
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        #invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], wh_y



    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        pu = nn.PixelUnshuffle(4)
        ps = nn.PixelShuffle(4)
        x = ps(x)
        x=torch.reshape(x,(B,C,H,W))
        y1,y2= self.forward_core(x)#first local scan
        y=y1+y2
        assert y1.dtype == torch.float32
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        #add
        y = y.permute(0, 3, 1, 2).contiguous()#getBCHW
        #torch.cuda.synchronize()
        #t3=time.time()
        yp=torch.reshape(y,(B,C//16,H*4,W*4))
        yp=pu(yp)#puback
        y3, y4 = self.forward_core2(yp)#second global scan
        assert y3.dtype == torch.float32
        y = y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        out = y * F.silu(z)
        return out,y

class MDSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0.,
            d_state: int = 16,
            expand: float = 2.,
            # is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        # copy
        self.ln_1_2 = norm_layer(hidden_dim // 2)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate,
                                   **kwargs)
        # copy
        # self.self_attention_2 = SS2D(d_model=hidden_dim // 2, d_state=d_state, expand=expand,
        #                              dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.drop_path_2 = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_2 = nn.Parameter(torch.ones(hidden_dim // 2))

        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale3 = nn.Parameter(torch.ones(hidden_dim))
        #
        self.l1 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, 1, 1, groups=4,bias=False)
        self.l2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 5, 1, 2, groups=4,bias=False)
        # self.l4 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, 1, 1)
        self.conv11 = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0,bias=False)

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input1 = input.view(B, *x_size, C).contiguous()
        X = self.ln_1(input1)
        # torch.cuda.synchronize()
        # stage1 = time.time()
        # torch.cuda.synchronize()
        # st = time.time()
        add,fs=self.self_attention(X)
        #x = input1 + self.self_attention(X)
        x = input1 + add
        mix = self.conv11(self.ln_2(x).permute(0, 3, 1, 2).contiguous())
        add2,fc=self.conv_blk(mix)
        x = x + add2.permute(0, 2, 3, 1).contiguous()
        # torch.cuda.synchronize()
        # ct1 = time.time()
        # print('Csupp$$$', ct1 - ct)
        # print(x.shape,'xxx')
        # z=x.view(B1,-1,C1)
        # print(z.shape,'z.shape')
        # print(x.view(B1,C1,H1,W1).shape,'***************************************')
        # X,2,128,128,90
        x = x.view(B, -1, C).contiguous()
        # x = x.view(B1, C1, H1, W1).contiguous()
        # X,[2,16384,90]
        return x,fs,fc
@ARCH_REGISTRY.register()
class Deblur(nn.Module):
    def __init__(self, num_feat=32, num_block=15):
        super().__init__()
        #########
        img_size = 128
        patch_size = 1
        in_chans = 192
        embed_dim = 96
        depths = (1,)
        drop_rate = 0.
        d_state = 16
        mlp_ratio = 1
        drop_path_rate = 0.
        norm_layer = nn.LayerNorm
        patch_norm = True
        use_checkpoint = False
        upscale = 2
        img_range = 1.
        upsampler = ''
        #######
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = embed_dim
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1,bias=False)  # xian,zai conv
        # self.conv_fir = nn.Conv2d(num_in_ch // 16, (embed_dim // 16) // 4, 3, 1, 1)  # xian,conv,zai
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = False
        # self.is_light_sr = True if self.upsampler == 'pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.norm = norm_layer(self.num_features)
        self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=4,bias=False),
                                             nn.SiLU(), nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=4,bias=False),
                                             nn.Conv2d(embed_dim, embed_dim, 1, padding=0,bias=False))
        # self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=1)
        # -------------------------3. high-quality image reconstruction ------------------------ #
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1,bias=False)
        self.apply(self._init_weights)
        self.s1 = ChannelAttention(num_feat=embed_dim)
        self.m = MDSSBlock(
            hidden_dim=embed_dim,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            d_state=d_state,
            expand=self.mlp_ratio
        )
        self.g1 = nn.SiLU()
        self.sig1 = nn.Sigmoid()

    def _init_weights(self, m):
        # print('***********init')
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # torch.cuda.synchronize()
        # ft= time.time()
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # N,L,C
        # x = self.pos_drop(x)
        x,fs,fc = self.m(x, x_size)
        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        x = self.sig1(x)
        # torch.cuda.synchronize()
        # ft1= time.time()
        # print('f-feature$$$', ft1-ft)
        return x,fs,fc

    def forward(self, x):
        b, t, c, h, w = x.size()
        old_shape = x.shape
        new_shape = (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = x.reshape(new_shape)#temporal pre-reshape
        pu = nn.PixelUnshuffle(4)
        ps = nn.PixelShuffle(4)
        x = pu(x)#spatial pre-reshape
        x_first = self.conv_first(x)
        x_first = self.g1(x_first)
        mid,fs,fc = self.forward_features(x_first)
        re = self.conv_after_body(mid) + mid
        x = x + self.conv_last(re)
        x=ps(x)
        x = x.reshape(old_shape)
        return x#,fs,fc




class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x



class PatchUnEmbed(nn.Module):
    r""" return D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)



