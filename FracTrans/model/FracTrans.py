
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Utility Functions: Fast Fractional Fourier Transform (FrFT) ---
def fast_frft_1d(x, alpha, dim=-1):
    alpha = alpha % 2
    phi = alpha * np.pi / 2
    if abs(phi % (2 * np.pi)) < 1e-8:
        return x
    n = x.shape[dim]
    scale = torch.sqrt(torch.tensor(n, dtype=torch.float32, device=x.device))
    t = (torch.arange(n, dtype=torch.float32, device=x.device) - n // 2) / scale
    t = t.view(*([1] * dim), -1, *([1] * (x.dim() - dim - 1)))
    cot_phi = torch.clamp(1 / torch.tan(phi), min=-1e4, max=1e4) if abs(torch.tan(phi)) > 1e-8 else torch.tensor(1e4, device=x.device)
    chirp_pre = torch.exp(-1j * np.pi * t ** 2 * cot_phi)
    x_pre = x * chirp_pre
    x_fft = torch.fft.fft(x_pre, dim=dim)
    chirp_post = torch.exp(-1j * np.pi * t ** 2 * cot_phi)
    x_frft = x_fft * chirp_post
    coeff = torch.sqrt((1 - 1j * cot_phi) / (2 * np.pi))
    x_frft = x_frft * coeff * scale
    x_frft_norm = torch.sqrt(torch.sum(torch.abs(x_frft) ** 2, dim=dim, keepdim=True)) + 1e-6
    x_frft = x_frft / x_frft_norm
    return x_frft.to(torch.complex64)

# --- Frequency Mode Selection ---
def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

# --- Patch Embedding ---
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=60, patch_size=4, dim_spatial=96):
        super().__init__()
        self.proj1 = nn.Conv2d(in_ch, dim_spatial // 2, kernel_size=patch_size, stride=patch_size)
        self.proj2 = nn.Conv2d(in_ch, dim_spatial // 2, kernel_size=patch_size * 2, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(dim_spatial)
        self.norm2 = nn.LayerNorm(dim_spatial)
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_uniform_(self.proj1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.proj2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = self.proj1(x)
        x2 = self.proj2(x)
        x2 = F.interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        x = self.norm(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm2(x)
        x = self.dropout(x)
        return x

# --- Patch Merging ---
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(2 * dim)
        self.norm2 = nn.LayerNorm(2 * dim)
        self.residual_conv = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.attn_pool = nn.Linear(2 * dim, 2 * dim)
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.residual_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.attn_pool.weight)

    def forward(self, x, h, w):
        B, N, D = x.shape
        assert N == h * w, f"PatchMerging input dimension mismatch: expected {h * w}, got {N}"
        x = x.view(B, h, w, D).permute(0, 3, 1, 2).contiguous()
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        residual = self.residual_conv(residual)
        residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + residual
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.attn_pool(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return x

class LocalChannelSpectralAttention(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=12):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = min(num_heads, window_size)
        if window_size % self.num_heads != 0:
            raise ValueError(f"window_size ({window_size}) must be divisible by num_heads ({self.num_heads})")
        self.dim_per_head = window_size // self.num_heads
        if dim % window_size != 0:
            raise ValueError(f"dim ({dim}) must be divisible by window_size ({window_size})")
        self.num_windows = dim // window_size
        self.shift_size = window_size // 2
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.query = nn.Linear(window_size, window_size)
        self.key = nn.Linear(window_size, window_size)
        self.value = nn.Linear(window_size, window_size)
        self.query_shift = nn.Linear(window_size, window_size)
        self.key_shift = nn.Linear(window_size, window_size)
        self.value_shift = nn.Linear(window_size, window_size)
        self.fusion = nn.Linear(2 * dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.3)
        )
        self.dropout = nn.Dropout(0.3)
        # 可学习的注意力参数
        self.attn_regular = nn.Parameter(torch.randn(self.num_windows, self.window_size))  # [num_windows, window_size]
        nn.init.xavier_uniform_(self.attn_regular, gain=1.414)
        # 修正：将 proj_attn 的输出维度改为 num_heads
        self.proj_attn = nn.Linear(window_size, self.num_heads)  # 输出 num_heads 而不是 dim_per_head
        nn.init.xavier_uniform_(self.proj_attn.weight)
        # 保存动态注意力权重
        self.dynamic_attn = None
        # 初始化其他权重
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.query_shift.weight)
        nn.init.xavier_uniform_(self.key_shift.weight)
        nn.init.xavier_uniform_(self.value_shift.weight)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[3].weight)

    def forward(self, x, h, w):
        B, N, D = x.shape
        assert N == h * w, f"LocalChannelSpectralAttention input size mismatch: expected {h * w}, got {N}"
        x = self.norm1(x)
        residual = x
        x_regular = x.reshape(B * N, self.num_windows, self.window_size)
        q_regular = self.query(x_regular).reshape(B * N, self.num_windows, self.num_heads, self.dim_per_head)
        k_regular = self.key(x_regular).reshape(B * N, self.num_windows, self.num_heads, self.dim_per_head)
        v_regular = self.value(x_regular).reshape(B * N, self.num_windows, self.num_heads, self.dim_per_head)
        # 计算动态注意力
        attn_regular = F.softmax((q_regular @ k_regular.transpose(-2, -1)) / (self.dim_per_head ** 0.5), dim=-1)  # [B * N, num_windows, num_heads, dim_per_head]
        logging.debug(f"attn_regular shape: {attn_regular.shape}")
        # 结合可学习参数
        learned_attn = torch.sigmoid(self.attn_regular)  # [num_windows, window_size]
        learned_attn = self.proj_attn(learned_attn)  # [num_windows, num_heads]
        learned_attn = learned_attn.view(1, self.num_windows, self.num_heads, 1)  # [1, num_windows, num_heads, 1]
        logging.debug(f"learned_attn shape: {learned_attn.shape}")
        attn_regular = attn_regular * learned_attn  # 广播融合 [B * N, num_windows, num_heads, dim_per_head]
        self.dynamic_attn = attn_regular  # 保存动态注意力权重
        attn_regular = self.dropout(attn_regular)
        x_regular = (attn_regular @ v_regular).reshape(B * N, self.num_windows * self.window_size)
        x_regular = x_regular.reshape(B, N, D)
        x_shifted = torch.roll(x, shifts=self.shift_size, dims=2)
        x_shift = x_shifted.reshape(B * N, self.num_windows, self.window_size)
        q_shift = self.query_shift(x_shift).reshape(B * N, self.num_windows, self.num_heads, self.dim_per_head)
        k_shift = self.key_shift(x_shift).reshape(B * N, self.num_windows, self.num_heads, self.dim_per_head)
        v_shift = self.value_shift(x_shift).reshape(B * N, self.num_windows, self.num_heads, self.dim_per_head)
        attn_shift = F.softmax((q_shift @ k_shift.transpose(-2, -1)) / (self.dim_per_head ** 0.5), dim=-1)
        attn_shift = self.dropout(attn_shift)
        x_shift = (attn_shift @ v_shift).reshape(B * N, self.num_windows * self.window_size)
        x_shift = x_shift.reshape(B, N, D)
        x_shift = torch.roll(x_shift, shifts=-self.shift_size, dims=2)
        x = torch.cat([x_regular, x_shift], dim=-1)
        x = self.fusion(x)
        x = x + residual
        x = self.norm2(x)
        residual_mlp = x
        x = self.mlp(x)
        x = x + residual_mlp
        return x

# --- Global Channel Spectral Attention with Dynamic Sparse Attention ---
class GlobalChannelSpectralAttention(nn.Module):
    def __init__(self, dim, num_clusters=32, num_heads=2):# 16 24 32 48 256
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.cluster_proj = nn.Linear(dim, num_clusters)
        self.cluster_center = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.3)
        )
        self.dropout = nn.Dropout(0.3)
        self.attn = None
        nn.init.xavier_uniform_(self.cluster_proj.weight)
        nn.init.xavier_uniform_(self.cluster_center.weight)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[3].weight)

    def forward(self, x, h, w):
        B, N, D = x.shape
        assert N == h * w, f"GlobalChannelSpectralAttention input size mismatch: expected {h * w}, got {N}"
        x = self.norm1(x)
        residual = x
        cluster_weights = F.softmax(self.cluster_proj(x), dim=-1)
        cluster_centers = self.cluster_center(x)
        cluster_features = torch.einsum('bnc,bnd->bcd', cluster_weights, cluster_centers)
        cluster_features = self.norm3(cluster_features)
        q = self.query(cluster_features).view(B, self.num_clusters, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.key(cluster_features).view(B, self.num_clusters, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.value(cluster_features).view(B, self.num_clusters, self.num_heads, self.dim_per_head).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) / (self.dim_per_head ** 0.5), dim=-1)
        self.attn = attn
        attn = self.dropout(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, self.num_clusters, D)
        x_out = torch.einsum('bnc,bcd->bnd', cluster_weights, x_attn)
        x = x_out + residual
        x = self.norm2(x)
        residual_mlp = x
        x = self.mlp(x)
        x = x + residual_mlp
        return x

# --- FFTransformer with Fourier Attention ---
class FFTransformer(nn.Module):
    def __init__(self, dim_spatial, num_heads=4, mlp_ratio=2.0, drop=0.3, modes=64, mode_select_method='random', activation='tanh'):
        super().__init__()
        self.dim_spatial = dim_spatial
        self.num_heads = num_heads
        self.modes = modes
        self.mode_select_method = mode_select_method
        self.activation = activation
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.conv1 = nn.Conv1d(dim_spatial, dim_spatial, kernel_size=1)
        self.norm1 = nn.LayerNorm(dim_spatial)
        self.query = nn.Linear(dim_spatial, dim_spatial)
        self.key = nn.Linear(dim_spatial, dim_spatial)
        self.value = nn.Linear(dim_spatial, dim_spatial)
        mlp_hidden_dim = int(dim_spatial * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(dim_spatial, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(mlp_hidden_dim, dim_spatial, 1),
            nn.Dropout(drop)
        )
        self.norm2 = nn.BatchNorm2d(dim_spatial)
        self.norm3 = nn.LayerNorm(dim_spatial)
        self.dropout = nn.Dropout(drop)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[3].weight)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, N, D = x.shape
        h = w = int(np.sqrt(N))
        residual = x.clone()
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.conv1(x).transpose(1, 2)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        qkv = torch.stack([q, k, v], dim=0)
        alpha_clamped = torch.clamp(self.alpha, 0.05, 0.95)
        qkv_ft = fast_frft_1d(qkv.to(torch.complex64), alpha_clamped, dim=2)
        q_ft, k_ft, v_ft = qkv_ft[0], qkv_ft[1], qkv_ft[2]
        index = get_frequency_modes(N, modes=self.modes, mode_select_method=self.mode_select_method)
        q_ft_ = q_ft[:, index, :]
        k_ft_ = k_ft[:, index, :]
        v_ft_ = v_ft[:, index, :]
        xqk_ft = torch.einsum("bxd,byd->bxy", q_ft_, k_ft_)
        # 缩放：防止复数 tanh 因内积幅度随 sqrt(D) 增大而过早饱和
        xqk_ft = xqk_ft / (self.dim_spatial ** 0.5)
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(xqk_ft.abs(), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        xqkv_ft = torch.einsum("bxy,byd->bxd", xqk_ft, v_ft_)
        out_ft = torch.zeros(B, N, D, device=x.device, dtype=torch.cfloat)
        for i, j in enumerate(index):
            out_ft[:, j, :] = xqkv_ft[:, i, :]
        x_attn = fast_frft_1d(out_ft, -alpha_clamped, dim=1).real
        x = x + x_attn
        x_mlp = self.mlp(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_mlp
        x = x.view(B, h, w, D).permute(0, 3, 1, 2)
        x = self.norm2(x).view(B, D, N).transpose(1, 2)
        x = self.norm3(x)
        x = self.dropout(x)
        x = x + residual
        return x

# --- Feature Processor ---
class FeatureProcessor(nn.Module):
    def __init__(self, dim_spatial, num_heads=4, window_size=12):
        super().__init__()
        self.local_attn = LocalChannelSpectralAttention(dim_spatial, num_heads=num_heads, window_size=window_size)
        self.global_attn = GlobalChannelSpectralAttention(dim_spatial, num_heads=num_heads)
        self.ffts = nn.ModuleList([FFTransformer(dim_spatial, num_heads=num_heads, drop=0.3, modes=64, mode_select_method='random', activation='tanh') for _ in range(2)])
        self.fusion = nn.Linear(dim_spatial * 2, dim_spatial)
        self.norm = nn.BatchNorm1d(dim_spatial)
        self.norm2 = nn.LayerNorm(dim_spatial)
        self.dropout = nn.Dropout(0.3)
        nn.init.xavier_uniform_(self.fusion.weight)

    def forward(self, x, h, w):
        B, N, D = x.shape
        assert N == h * w, f"FeatureProcessor input size mismatch: expected {h * w}, got {N}"
        residual = x.clone()
        x_local = self.local_attn(x, h, w)
        x_local = x_local + residual
        x_fft1 = self.ffts[0](x_local)
        x_global = self.global_attn(x_local, h, w)
        x_global = x_global + x_local
        x_fft2 = self.ffts[1](x_global)
        x = self.fusion(torch.cat([x_fft1, x_fft2], dim=-1))
        x = self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x = self.norm2(x)
        x = self.dropout(x)
        return x + residual

# --- Adaptive Pooling ---
class AdaptivePooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, C, H, W = x.shape
        weights = self.attn(x.flatten(2).transpose(1, 2).contiguous())
        weights = self.sigmoid(weights).transpose(1, 2).reshape(B, C, H, W).contiguous()
        weights = self.dropout(weights)
        x_weighted = x * weights
        x_pooled = F.adaptive_avg_pool2d(x_weighted, (1, 1)).flatten(1)
        return x_pooled

# --- Main Model Class ---
class SpatialFractionalTransformerSwin(nn.Module):
    def __init__(self, in_channels=60, num_classes=2, image_size=256, dim_spatial=96, num_heads=4, depth=3, window_size=12):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.dim_spatial = dim_spatial
        self.depth = depth
        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("std", torch.ones(in_channels))
        self.patch_embed = PatchEmbed(in_ch=in_channels, patch_size=4, dim_spatial=dim_spatial)
        self.stages = nn.ModuleList()
        current_dim = dim_spatial
        for i in range(depth):
            stage = nn.ModuleList([
                FeatureProcessor(current_dim, num_heads=num_heads, window_size=window_size),
                PatchMerging(current_dim) if i < depth - 1 else nn.Identity()
            ])
            self.stages.append(stage)
            if i < depth - 1:
                current_dim *= 2
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(current_dim // (2 ** (depth - 1 - i)), dim_spatial, kernel_size=1) for i in range(depth)
        ])
        self.upsample_convs = nn.ModuleList([
            nn.Conv2d(dim_spatial, dim_spatial, kernel_size=1) for _ in range(depth - 1)
        ])
        self.pooling = AdaptivePooling(dim_spatial)
        self.cls_head = nn.Sequential(
            nn.Linear(dim_spatial, dim_spatial),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim_spatial, dim_spatial // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim_spatial // 2, dim_spatial // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim_spatial // 4, num_classes)
        )
        self.aux_heads = nn.ModuleList(
            [nn.Linear(current_dim // (2 ** (depth - 1 - 2 * i)), num_classes) for i in range(depth // 2)])
        nn.init.xavier_uniform_(self.cls_head[0].weight)
        nn.init.xavier_uniform_(self.cls_head[3].weight)
        nn.init.xavier_uniform_(self.cls_head[6].weight)
        nn.init.xavier_uniform_(self.cls_head[9].weight)
        for layer in self.cls_head:
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for aux_head in self.aux_heads:
            nn.init.xavier_uniform_(aux_head.weight)
            if aux_head.bias is not None:
                nn.init.zeros_(aux_head.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, f"Input size mismatch: expected ({self.image_size}, {self.image_size}), got ({H}, {W})"
        x = (x - self.mean.view(1, -1, 1, 1)) / (self.std.view(1, -1, 1, 1) + 1e-6)
        x_tokens = self.patch_embed(x)
        stage_outputs = []
        current_h, current_w = H // 4, W // 4
        for i, stage in enumerate(self.stages):
            processor, patch_merge = stage
            x_processed = processor(x_tokens, current_h, current_w)
            stage_outputs.append(x_processed.view(B, current_h, current_w, -1).contiguous())
            if i < self.depth - 1:
                x_tokens = patch_merge(x_tokens, current_h, current_w)
                current_h //= 2
                current_w //= 2
        p_last = self.fpn_convs[-1](stage_outputs[-1].permute(0, 3, 1, 2))
        for i in range(len(stage_outputs) - 2, -1, -1):
            p_upsampled = F.interpolate(p_last, size=(stage_outputs[i].shape[1], stage_outputs[i].shape[2]), mode='bilinear', align_corners=False)
            p_last = self.fpn_convs[i](stage_outputs[i].permute(0, 3, 1, 2)) + self.upsample_convs[i](p_upsampled)
        x_pooled = self.pooling(p_last)
        residual = x_pooled
        x_cls = self.cls_head(x_pooled)
        x_cls = x_cls + residual[:, :self.num_classes]
        logits = x_cls
        aux_logits = [aux_head(stage_outputs[2 * i].mean(dim=(1, 2))) for i, aux_head in enumerate(self.aux_heads)]
        return logits, aux_logits

    def get_frft_params(self):
        frft_params = []
        for stage in self.stages:
            processor = stage[0]
            for fft in processor.ffts:
                frft_params.append(fft.alpha)
        return frft_params

# --- Convenience Function ---
def get_SFT_Swin(in_channels=60, num_classes=2, image_size=256, activation='tanh'):
    return SpatialFractionalTransformerSwin(
        in_channels=in_channels,
        num_classes=num_classes,
        image_size=image_size,
        dim_spatial=96,
        num_heads=4,
        depth=3,
        window_size=12
    )
def get_SFT_PLGC_Swin(in_channels=40, num_classes=3, image_size=256, activation='tanh'):
    return SpatialFractionalTransformerSwin(
        in_channels=in_channels,
        num_classes=num_classes,
        image_size=image_size,
        dim_spatial=96,
        num_heads=4,
        depth=3,
        window_size=12
    )

# --- Test Code ---
if __name__ == "__main__":
    model = get_SFT_Swin()
    model.eval().to(device)
    x = torch.randn(1, 60, 256, 256).to(device)
    with torch.no_grad(), autocast():
        logits, aux_logits = model(x)
    print(f"Main output shape: {logits.shape}")
    print(f"Number of aux outputs: {len(aux_logits)}")
    for i, aux_logit in enumerate(aux_logits):
        print(f"Aux output {i} shape: {aux_logit.shape}")
