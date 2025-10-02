# ProVAE.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic Blocks
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.body(x)
        x = F.avg_pool2d(x, 2)  # downsample by 2
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # upsample by 2
        x = self.body(x)
        return x

# -----------------------------
# Progressive Encoder (shared)
# -----------------------------
class ProgressiveEncoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32, base_ch=128, max_stage=3):
        super().__init__()
        self.in_ch = in_ch
        self.z_dim = z_dim
        self.max_stage = max_stage
        ch = [base_ch] * (max_stage + 1)
        
        self.from_rgb = nn.ModuleList([nn.Conv2d(in_ch, ch[s], 1) for s in range(max_stage + 1)])
        self.blocks = nn.ModuleList([])
        for s in range(max_stage, 0, -1):
            self.blocks.append(DownBlock(ch[s], ch[s-1]))
        
        self.final_conv = ConvBlock(ch[0], ch[0])
        self.mu = nn.Linear(ch[0] * 4 * 4, z_dim)
        self.logvar = nn.Linear(ch[0] * 4 * 4, z_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x, stage: int, alpha: float = 1.0):
        assert 0 <= stage <= self.max_stage
        
        if stage == 0:
            h = self.from_rgb[0](x)
        else:
            # --- 修正点 1: 修复 Encoder 的 fade-in 逻辑 ---
            # 旧路径 (低分辨率)
            x_down = F.avg_pool2d(x, 2)
            h_old = self.from_rgb[stage - 1](x_down)
            
            # 新路径 (高分辨率)
            h_new = self.from_rgb[stage](x)
            
            # 将新路径的特征图通过一个下采样块，使其分辨率与旧路径对齐
            # block 的索引是从大 stage 到小 stage，所以 max_stage - stage 对应正确的块
            # 例如 stage=3 (32x32), block_idx=0, 对应 ch[3]->ch[2] 的 DownBlock
            block_idx = self.max_stage - stage
            h_new_processed = self.blocks[block_idx](h_new)
            
            # 在低分辨率特征空间进行混合
            h = (1 - alpha) * h_old + alpha * h_new_processed
            # --- 修正结束 ---

        # 从混合后的特征继续下采样到 4x4
        # h 此时的分辨率等同于 stage-1 级别
        current_effective_stage = stage - 1 if stage > 0 else 0
        
        # 从当前有效分辨率下采样到 4x4
        # 例如 current_effective_stage=2 (16x16), 循环 s 将是 2, 1
        for s in range(current_effective_stage, 0, -1):
            # s=2 (16x16->8x8), block_idx=max-2
            # s=1 (8x8->4x4), block_idx=max-1
            block_idx = self.max_stage - s
            h = self.blocks[block_idx](h)
        
        h = self.final_conv(h)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

# -----------------------------
# Progressive Decoder (Fade-in)
# -----------------------------
class ProgressiveDecoderFadeIn(nn.Module):
    def __init__(self, z_dim=32, out_ch=1, base_ch=128, max_stage=3):
        super().__init__()
        self.z_dim = z_dim
        self.max_stage = max_stage
        ch = [base_ch] * (max_stage + 1)
        self.fc = nn.Linear(z_dim, ch[0] * 4 * 4)
        self.block_up = nn.ModuleList([])
        for s in range(0, max_stage):
            self.block_up.append(UpBlock(ch[s], ch[s+1]))
        self.to_rgb = nn.ModuleList([nn.Conv2d(ch[s], out_ch, 1) for s in range(max_stage + 1)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z, stage: int, alpha: float = 1.0):
        assert 0 <= stage <= self.max_stage
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        
        if stage == 0:
            return torch.sigmoid(self.to_rgb[0](h))

        # 构造到上一层的特征 h_prev，但不包括最后一层 up_block
        h_prev = h
        for s in range(stage - 1):
            h_prev = self.block_up[s](h_prev)
        
        # 旧路径: 上一层的 RGB 输出，然后上采样
        rgb_prev = self.to_rgb[stage - 1](h_prev)
        rgb_prev_up = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')

        # 新路径: 当前层的特征和 RGB 输出
        h_cur = self.block_up[stage - 1](h_prev)
        rgb_cur = self.to_rgb[stage](h_cur)

        return torch.sigmoid((1 - alpha) * rgb_prev_up + alpha * rgb_cur)

# -----------------------------
# Progressive Decoder (Residual)
# -----------------------------
class ProgressiveDecoderResidual(nn.Module):
    def __init__(self, z_dim=32, out_ch=1, base_ch=128, max_stage=3):
        super().__init__()
        self.z_dim = z_dim
        self.max_stage = max_stage
        ch = [base_ch] * (max_stage + 1)
        self.fc = nn.Linear(z_dim, ch[0] * 4 * 4)
        self.block_up = nn.ModuleList([])
        for s in range(0, max_stage):
            self.block_up.append(UpBlock(ch[s], ch[s+1]))
        self.to_rgb = nn.ModuleList([nn.Conv2d(ch[s], out_ch, 1) for s in range(max_stage + 1)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z, stage: int):
        assert 0 <= stage <= self.max_stage
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        img = torch.sigmoid(self.to_rgb[0](h))
        
        if stage == 0:
            return img
            
        h_cur = h
        for s in range(stage):
            h_cur = self.block_up[s](h_cur)
            res_s = self.to_rgb[s + 1](h_cur)
            img = F.interpolate(img, scale_factor=2, mode='nearest') + torch.sigmoid(res_s)
            
        return img

# -----------------------------
# ProVAE (Fade-in) & Res-ProVAE
# -----------------------------
class Reparameterize(nn.Module):
    def forward(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

class ProVAE_FadeIn(nn.Module):
    def __init__(self, in_ch=1, z_dim=32, base_ch=128, max_stage=3):
        super().__init__()
        self.encoder = ProgressiveEncoder(in_ch=in_ch, z_dim=z_dim, base_ch=base_ch, max_stage=max_stage)
        self.decoder = ProgressiveDecoderFadeIn(z_dim=z_dim, out_ch=in_ch, base_ch=base_ch, max_stage=max_stage)
        self.reparam = Reparameterize()

    def forward(self, x, stage: int, alpha: float):
        mu, logvar = self.encoder(x, stage, alpha)
        z = self.reparam(mu, logvar)
        x_recon = self.decoder(z, stage, alpha)
        return x_recon, mu, logvar

class Res_ProVAE(nn.Module):
    def __init__(self, in_ch=1, z_dim=32, base_ch=128, max_stage=3):
        super().__init__()
        self.encoder = ProgressiveEncoder(in_ch=in_ch, z_dim=z_dim, base_ch=base_ch, max_stage=max_stage)
        self.decoder = ProgressiveDecoderResidual(z_dim=z_dim, out_ch=in_ch, base_ch=base_ch, max_stage=max_stage)
        self.reparam = Reparameterize()

    def forward(self, x, stage: int, alpha: float = 1.0):
        # 编码端仍按当前 stage 处理；为保持接口一致性，alpha 强制为 1.0
        # 残差解码器不需要 alpha
        mu, logvar = self.encoder(x, stage, alpha=alpha)
        z = self.reparam(mu, logvar)
        x_recon = self.decoder(z, stage)
        return x_recon, mu, logvar

# -----------------------------
# Loss
# -----------------------------
def vae_loss(x, x_recon, mu, logvar):
    """VAE损失函数，与VAE模型保持一致的命名和形式"""
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD