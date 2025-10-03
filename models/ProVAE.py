import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    - enc: 3x3 Conv + LeakyReLU + AvgPool(2)
    - dec: Upsample(2) + 3x3 Conv + LeakyReLU
    """
    def __init__(self, in_ch, out_ch, mode: str):
        super().__init__()
        assert mode in ("enc", "dec")
        self.mode = mode
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2)
        if mode == "enc":
            self.down = nn.AvgPool2d(2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        if self.mode == "enc":
            x = self.conv(x)
            x = self.act(x)
            x = self.down(x)
            return x
        else:
            x = self.up(x)
            x = self.conv(x)
            x = self.act(x)
            return x


class ProVAE(nn.Module):
    """
    Progressive VAE
    - start_res -> final_res（2 的幂增长）
    - stage 表示当前层级（0 对应 start_res）
    - alpha 为 fade-in 权重
    """
    def __init__(
        self,
        in_ch: int = 1,
        z_dim: int = 20,
        start_res: int = 4,
        final_res: int = 32,
        base_ch: int = 128,
        min_ch: int = 16,
    ):
        super().__init__()
        assert final_res % start_res == 0 and (final_res & (final_res - 1)) == 0
        
        self.in_ch = in_ch
        self.z_dim = z_dim
        self.start_res = start_res
        self.final_res = final_res

        # levels: [start_res, start_res*2, ..., final_res]
        self.levels = int(math.log2(final_res // start_res)) + 1  # 例如 4->32: levels=4 (4/8/16/32)
        self.max_stage = self.levels - 1

        # 通道数从低分辨率到高分辨率逐步变小
        chs = []
        for i in range(self.levels):
            ch = max(base_ch // (2 ** i), min_ch)
            chs.append(ch)
        self.chs = chs  # 例如 [128, 64, 32, 16]

        # -------- Encoder --------
        self.enc_from_rgb = nn.ModuleList([nn.Conv2d(in_ch, c, 1) for c in self.chs])
        # enc_blocks: level i -> i-1 (i 从高到低)，降采样一半
        self.enc_blocks = nn.ModuleList([
            ConvBlock(self.chs[i], self.chs[i-1], mode="enc") for i in range(1, self.levels)
        ])
        self.enc_base = nn.Conv2d(self.chs[0], self.chs[0], 3, 1, 1)

        flat_dim = self.chs[0] * self.start_res * self.start_res
        self.fc_mu = nn.Linear(flat_dim, z_dim)
        self.fc_logvar = nn.Linear(flat_dim, z_dim)

        # -------- Decoder --------
        self.fc_decode = nn.Linear(z_dim, flat_dim)
        # dec_blocks: level i-1 -> i (i 从低到高)，上采样一倍
        self.dec_blocks = nn.ModuleList([
            ConvBlock(self.chs[i-1], self.chs[i], mode="dec") for i in range(1, self.levels)
        ])
        self.dec_to_rgb = nn.ModuleList([nn.Conv2d(c, in_ch, 1) for c in self.chs])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --- utils ---
    def _res_of(self, stage: int) -> int:
        return self.start_res * (2 ** stage)

    # --- Encoder with fade-in ---
    def encode(self, x, stage: int, alpha: float):
        assert 0 <= stage <= self.max_stage
        cur_res = self._res_of(stage)
        
        # 从最高分辨率开始，逐步降采样并处理
        h = self.enc_from_rgb[stage](F.interpolate(x, size=(cur_res, cur_res), mode="area"))

        if stage > 0:
            h = self.enc_blocks[stage - 1](h)
            low_res = self._res_of(stage - 1)
            low_h = self.enc_from_rgb[stage - 1](F.interpolate(x, size=(low_res, low_res), mode="area"))
            h = alpha * h + (1.0 - alpha) * low_h

            for s in range(stage - 1, 0, -1):
                h = self.enc_blocks[s - 1](h)

        h = F.relu(self.enc_base(h), inplace=True)

        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    # --- Decoder with fade-in ---
    def decode(self, z, stage: int, alpha: float):
        assert 0 <= stage <= self.max_stage
        h = self.fc_decode(z)
        h = h.view(-1, self.chs[0], self.start_res, self.start_res)
        h = F.relu(h, inplace=True)

        if stage == 0:
            rgb = self.dec_to_rgb[0](h)
            return torch.sigmoid(rgb)

        prev_rgb = self.dec_to_rgb[0](h)  # base 的 toRGB (start_res)
        for s in range(1, stage + 1):
            h = self.dec_blocks[s - 1](h)  # 上到下一层
            cur_rgb = self.dec_to_rgb[s](h)
            if s == stage:
                up_prev = F.interpolate(prev_rgb, scale_factor=2, mode="nearest")
                rgb = alpha * cur_rgb + (1.0 - alpha) * up_prev
            else:
                prev_rgb = cur_rgb

        return torch.sigmoid(rgb)

    def forward(self, x, stage: int, alpha: float):
        mu, logvar = self.encode(x, stage=stage, alpha=alpha)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, stage=stage, alpha=alpha)
        return x_recon, mu, logvar
