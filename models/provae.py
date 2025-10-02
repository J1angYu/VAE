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
    """
    stage 0 -> 4x4, stage 1 -> 8x8, ... resolution = 4 * 2**stage
    通道安排保持简单，MNIST 到 32x32 足够。
    fade-in 在输入端：新路径 (x -> 新块) 与旧路径 (downsample(x) -> 旧encoder) 线性混合。
    """
    def __init__(self, in_ch=1, z_dim=32, base_ch=128, max_stage=3):
        super().__init__()
        self.in_ch = in_ch
        self.z_dim = z_dim
        self.max_stage = max_stage  # 0..3 对应 4,8,16,32
        # 每个 stage 的特征通道（简单起见固定）
        ch = [base_ch, base_ch, base_ch, base_ch]
        self.from_rgb = nn.ModuleList([nn.Conv2d(in_ch, ch[s], 1) for s in range(max_stage+1)])
        self.blocks = nn.ModuleList([])  # 自 stage s 向下到 4x4 的下采样块
        for s in range(max_stage, 0, -1):
            self.blocks.append(DownBlock(ch[s], ch[s-1]))
        # 4x4 顶部
        self.final_conv = ConvBlock(ch[0], ch[0])
        self.mu = nn.Linear(ch[0]*4*4, z_dim)
        self.logvar = nn.Linear(ch[0]*4*4, z_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x, stage:int, alpha:float=1.0):
        # x: B×1×R×R, stage∈[0, max_stage]
        assert 0 <= stage <= self.max_stage
        
        if stage == 0:
            # Stage 0: 直接处理 4x4 输入
            h = self.from_rgb[0](x)
        else:
            # new path（当前分辨率）
            h_new = self.from_rgb[stage](x)
            
            # old path（先下采样一半，再走上一个 stage 的 from_rgb）
            x_down = F.avg_pool2d(x, 2)
            h_old = self.from_rgb[stage-1](x_down)
            
            # 混合两个路径 - 此时 h_new 和 h_old 分辨率不同，需要先让 h_new 下采样
            # h_new 需要下采样到与 h_old 相同的分辨率
            h_new_down = F.avg_pool2d(h_new, 2)
            h = (1 - alpha) * h_old + alpha * h_new_down
        
        # 从混合后的特征继续下采样到 4x4
        # 计算当前 h 的 stage（对于 stage > 0，h 现在相当于 stage-1 的分辨率）
        current_stage = stage - 1 if stage > 0 else 0
        
        # 从当前分辨率下采样到 4x4
        for i in range(current_stage, 0, -1):
            idx = self.max_stage - i
            h = self.blocks[idx](h)
        
        h = self.final_conv(h)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

# -----------------------------
# Progressive Decoder (Fade-in)
# -----------------------------
class ProgressiveDecoderFadeIn(nn.Module):
    """
    解码端渐进 + fade-in：
      - old_rgb: 上一分辨率输出上采样到当前分辨率
      - new_rgb: 通过新 UpBlock 后的 to_rgb
      - 混合: (1 - alpha) * old_rgb + alpha * new_rgb
    """
    def __init__(self, z_dim=32, out_ch=1, base_ch=128, max_stage=3):
        super().__init__()
        self.z_dim = z_dim
        self.max_stage = max_stage
        ch = [base_ch, base_ch, base_ch, base_ch]  # 4->8->16->32
        self.fc = nn.Linear(z_dim, ch[0]*4*4)
        self.block_up = nn.ModuleList([])
        for s in range(0, max_stage):
            self.block_up.append(UpBlock(ch[s], ch[s+1]))
        self.to_rgb = nn.ModuleList([nn.Conv2d(ch[s], out_ch, 1) for s in range(max_stage+1)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z, stage:int, alpha:float=1.0):
        assert 0 <= stage <= self.max_stage
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        if stage == 0:
            return self.to_rgb[0](h)  # logits

        # 构造到上一层的特征
        h_prev = h
        for s in range(0, stage):
            if s == stage - 1:
                break
            h_prev = self.block_up[s](h_prev)
        rgb_prev = self.to_rgb[stage-1](h_prev)
        rgb_prev_up = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')

        # 当前层的新路径
        h_cur = self.block_up[stage-1](h_prev)
        rgb_cur = self.to_rgb[stage](h_cur)

        return (1 - alpha) * rgb_prev_up + alpha * rgb_cur  # logits

# -----------------------------
# Progressive Decoder (Residual)
# -----------------------------
class ProgressiveDecoderResidual(nn.Module):
    """
    拉普拉斯/残差式解码：
      - stage 0 直接产出 4×4 的基础图像 logits
      - 更高 stage：把上一分辨率图像上采样 ×2，再加上当前 stage 预测的 residual
      - 最终得到当前分辨率的 logits
    """
    def __init__(self, z_dim=32, out_ch=1, base_ch=128, max_stage=3):
        super().__init__()
        self.z_dim = z_dim
        self.max_stage = max_stage
        ch = [base_ch, base_ch, base_ch, base_ch]
        self.fc = nn.Linear(z_dim, ch[0]*4*4)
        self.block_up = nn.ModuleList([])
        for s in range(0, max_stage):
            self.block_up.append(UpBlock(ch[s], ch[s+1]))
        self.to_rgb = nn.ModuleList([nn.Conv2d(ch[s], out_ch, 1) for s in range(max_stage+1)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z, stage:int):
        assert 0 <= stage <= self.max_stage
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        img_logits = self.to_rgb[0](h)   # base at 4x4
        if stage == 0:
            return img_logits
        h_cur = h
        for s in range(0, stage):
            h_cur = self.block_up[s](h_cur)
            res_s = self.to_rgb[s+1](h_cur)  # residual at this scale
            img_logits = F.interpolate(img_logits, scale_factor=2, mode='nearest') + res_s
        return img_logits

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

    def forward(self, x, stage:int, alpha:float):
        mu, logvar = self.encoder(x, stage, alpha)
        z = self.reparam(mu, logvar)
        logits = self.decoder(z, stage, alpha)
        return logits, mu, logvar

class Res_ProVAE(nn.Module):
    def __init__(self, in_ch=1, z_dim=32, base_ch=128, max_stage=3):
        super().__init__()
        self.encoder = ProgressiveEncoder(in_ch=in_ch, z_dim=z_dim, base_ch=base_ch, max_stage=max_stage)
        self.decoder = ProgressiveDecoderResidual(z_dim=z_dim, out_ch=in_ch, base_ch=base_ch, max_stage=max_stage)
        self.reparam = Reparameterize()

    def forward(self, x, stage:int, alpha:float=None):
        # 编码端仍按当前 stage 处理；残差解码器不需要 alpha
        mu, logvar = self.encoder(x, stage, alpha=1.0)
        z = self.reparam(mu, logvar)
        logits = self.decoder(z, stage)
        return logits, mu, logvar

# -----------------------------
# Loss
# -----------------------------
def elbo_bce_logits(x, logits, mu, logvar):
    # x: [0,1], logits: 任意实数
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl
