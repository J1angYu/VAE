import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 基础卷积块：与 ProVAE 风格一致
# enc: 3x3 Conv + LeakyReLU + AvgPool(2)
# dec: Upsample(2) + 3x3 Conv + LeakyReLU
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mode: str):
        super().__init__()
        assert mode in ("enc", "dec")
        self.mode = mode
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.mode == "enc":
            x = self.conv(x)
            x = self.act(x)
            # Progressive 判别器/编码器常用均值下采样，稳定分布
            x = F.avg_pool2d(x, 2)
            return x
        else:
            # 生成器/解码器：最近邻上采样更接近 PGAN 做法
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.conv(x)
            x = self.act(x)
            return x


# ------------------------------------------
# Residual Progressive VAE
# - 每个 stage 独立学习该分辨率的残差 ΔX_s
# - s=0 直接输出基底图像（Sigmoid）
# - s>0 输出线性残差（不加激活），与上一级重建相加
# - 新增 stage 采用 fade-in：alpha 在 [0,1]
# ------------------------------------------
class ResidualProgressiveVAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        start_res: int = 4,
        final_res: int = 32,
        latent_dim: int = 128,
        base_ch: int = 128,
        min_ch: int = 16,
    ):
        super().__init__()
        assert final_res % start_res == 0 and ((final_res // start_res) & ((final_res // start_res) - 1)) == 0, \
            "final_res 必须是 start_res 的 2 的幂次倍数，例如 4->32"

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.start_res = start_res
        self.final_res = final_res
        self.latent_dim = latent_dim

        # levels: [start_res, start_res*2, ..., final_res]
        self.levels = int(math.log2(final_res // start_res)) + 1  # 例如 4->32: levels=4 (4/8/16/32)
        self.max_stage = self.levels - 1

        # 通道数从低分辨率到高分辨率逐步变小（贴近 PGAN 通道调度）
        chs = []
        for i in range(self.levels):
            ch = max(base_ch // (2 ** i), min_ch)
            chs.append(ch)
        self.chs = chs  # 例如 [128, 64, 32, 16]

        # ---------------- Encoder ----------------
        # fromRGB: 每个分辨率一层 1x1 将图像映射到特征
        self.enc_from_rgb = nn.ModuleList([nn.Conv2d(in_ch, c, 1) for c in self.chs])

        # 多级下采样块：s->s-1（s>=1）。数量 = levels-1
        self.enc_blocks = nn.ModuleList()
        for s in range(1, self.levels):
            self.enc_blocks.append(ConvBlock(self.chs[s], self.chs[s - 1], mode="enc"))

        # 最底层的“base”卷积，保证所有 stage（包括 s=0）都会经过
        self.enc_base = nn.Conv2d(self.chs[0], self.chs[0], 3, 1, 1)

        # 每个 stage 自己的 μ、logvar 全连接头（从 base feature flatten 后得到）
        flat_dim = self.chs[0] * self.start_res * self.start_res
        self.fc_mu = nn.ModuleList([nn.Linear(flat_dim, latent_dim) for _ in range(self.levels)])
        self.fc_logvar = nn.ModuleList([nn.Linear(flat_dim, latent_dim) for _ in range(self.levels)])

        # ---------------- Decoder ----------------
        # 每个 stage 自己的 z->base feature 线性层（独立 VAE 语义更清晰）
        self.fc_z_to_base = nn.ModuleList([nn.Linear(latent_dim, flat_dim) for _ in range(self.levels)])

        # 多级上采样块：0->1->...->(levels-1)（共享给所有 stage 使用）
        self.dec_blocks = nn.ModuleList()
        for s in range(0, self.levels - 1):
            self.dec_blocks.append(ConvBlock(self.chs[s], self.chs[s + 1], mode="dec"))

        # toResidual：每个分辨率一层 1x1 输出
        # - s=0: 作为基底图像输出，使用 sigmoid（[0,1]）
        # - s>0: 作为残差 ΔX_s 输出（线性，不加激活），用于回归残差
        self.to_residual = nn.ModuleList([nn.Conv2d(self.chs[s], out_ch, 1) for s in range(self.levels)])

        # 初始化（可按需添加更复杂的初始化）
        self._init_weights()

    # -------------- 工具函数 --------------
    def _res_of(self, stage: int) -> int:
        return self.start_res * (2 ** stage)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def reparameterize(mu, logvar):
        # 经典 reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -------------- 编码器（单个 stage）--------------
    def _encode_stage(self, x_residual, stage: int, alpha: float):
        """
        x_residual: 当前分辨率的残差目标（X - up(recon_{stage-1}))，s=0 时就是 X
        stage: 当前要编码的分辨率 stage
        alpha: fade-in 系数（仅在“引入新层的过渡期”用于 s 与 s-1 的混合）
        """
        cur_res = self._res_of(stage)
        x_cur = F.interpolate(x_residual, size=(cur_res, cur_res), mode="area")

        # fromRGB at current stage
        h = self.enc_from_rgb[stage](x_cur)

        if stage > 0:
            # 新层路径：当前分辨率卷积后下采样一步（s -> s-1）
            h_new = self.enc_blocks[stage - 1](h)

            # 旧路径：直接用低一档分辨率的 fromRGB（相当于 PGAN 判别器过渡期）
            low_res = self._res_of(stage - 1)
            x_low = F.interpolate(x_residual, size=(low_res, low_res), mode="area")
            h_old = self.enc_from_rgb[stage - 1](x_low)

            # 两路在同一分辨率（s-1）进行 fade-in 混合
            h = alpha * h_new + (1.0 - alpha) * h_old

            # 继续往下到 base
            for s in range(stage - 1, 0, -1):
                h = self.enc_blocks[s - 1](h)

        # 所有 stage（包括 0）都走 enc_base，确保底层表达一致
        h = F.leaky_relu(self.enc_base(h), negative_slope=0.2, inplace=True)

        # flatten -> μ, logvar
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu[stage](h_flat)
        logvar = self.fc_logvar[stage](h_flat)
        return mu, logvar

    # -------------- 解码器（单个 stage）--------------
    def _decode_stage(self, z, stage: int):
        """
        将 z_s 解码到第 stage 个分辨率的特征，并输出：
        - s=0: 基底图像 X'_0（Sigmoid）
        - s>0: 残差 ΔX_s（线性）
        """
        # z -> base feature
        feat = self.fc_z_to_base[stage](z)
        feat = feat.view(z.size(0), self.chs[0], self.start_res, self.start_res)
        feat = F.leaky_relu(feat, negative_slope=0.2, inplace=True)

        # 逐层上采样到目标分辨率
        for s in range(0, stage):
            feat = self.dec_blocks[s](feat)

        # 输出层
        out = self.to_residual[stage](feat)
        if stage == 0:
            # s=0 直接生成基底图像（[0,1]）
            out = torch.sigmoid(out)
        # s>0 时为线性残差（不做激活），用于 Gaussian/MSE 残差回归
        return out

    # -------------- 前向：从 s=0 到当前 stage 逐步 refine --------------
    def forward(self, x, stage: int, alpha: float = 1.0, detach_prev: bool = True):
        """
        x:  输入图像（任意分辨率，内部会按当前 stage 的分辨率对齐）
        stage: 当前训练/推理的最高 stage（0..max_stage）
        alpha: fade-in 系数；在每次从 stage-1 过渡到 stage 时使用
        detach_prev: 计算下一层残差时，是否对上一层重建做 detach（推荐 True，避免梯度穿透使低层被高层牵制）
        返回：
          recon:    当前 stage 的最终重建
          mu_list:  [μ_0, ..., μ_stage]
          logvar_list: 同上
          extras:   便于可视化/调试的中间结果（按需）
        """
        assert 0 <= stage <= self.max_stage

        # 起始：recon_(-1) 定义为 0（全黑）
        recon = torch.zeros_like(F.interpolate(x, size=(self._res_of(0), self._res_of(0)), mode="area"))

        mu_list, logvar_list = [], []
        deltas, recons = [], []

        for s in range(0, stage + 1):
            cur_res = self._res_of(s)
            x_cur = F.interpolate(x, size=(cur_res, cur_res), mode="area")

            # 计算当前残差目标：D_s = X_cur - up(recon_{s-1})
            prev_up = F.interpolate(recon, size=(cur_res, cur_res), mode="nearest")
            prev_up_detached = prev_up.detach() if detach_prev else prev_up
            residual_target = x_cur - prev_up_detached  # 这是要被 VAE_s 学习的目标

            # 编码当前残差（使用 fade-in）
            mu, logvar = self._encode_stage(residual_target, stage=s, alpha=(alpha if s == stage else 1.0))
            z = self.reparameterize(mu, logvar)

            # 解码得到 ΔX_s 或基底 X'_0
            out = self._decode_stage(z, stage=s)

            # 用 ΔX_s 更新重建；并做 fade-in（仅当前过渡 stage 使用 alpha）
            if s == 0:
                # 直接是基底图像
                cur_recon_candidate = out
                recon = cur_recon_candidate
            else:
                cur_recon_candidate = prev_up + out  # up(recon_{s-1}) + ΔX_s
                a = (alpha if s == stage else 1.0)
                recon = a * cur_recon_candidate + (1.0 - a) * prev_up

            # 记录
            mu_list.append(mu)
            logvar_list.append(logvar)
            deltas.append(out)
            recons.append(recon)

        # 为可视化友好，可对最终重建 clamp 到 [0,1]
        recon_vis = torch.clamp(recon, 0.0, 1.0)
        extras = {
            "deltas": deltas,           # [ΔX_0(or X0'), ΔX_1, ..., ΔX_s]
            "recons": recons,           # [X̂_0, X̂_1, ..., X̂_s]
            "resolutions": [self._res_of(i) for i in range(stage + 1)]
        }
        return recon_vis, mu_list, logvar_list, extras


# ------------------------------------------
# 参考损失函数（Residual Progressive 训练）
# - MNIST: s=0 用 BCE(X vs X'_0)，s>0 用 MSE( D_s vs ΔX_s )
# - FreyFace: 所有 stage 都用 Gaussian/MSE
# ------------------------------------------
def residual_pvae_loss(
    x,
    recon_vis,           # forward 返回的最终重建（仅用于可选的整体重建监控）
    mu_list, logvar_list,
    extras,
    dataset: str = "mnist",
    beta: float = 1.0,
    reduction: str = "mean",
):
    """
    返回：total_loss, dict(逐项损失)
    说明：
      - s=0 的监督对象：X_0（按 s=0 分辨率重采样） vs X'_0（Sigmoid 输出）
      - s>0 的监督对象：D_s = X_s - up(X̂_{s-1})  vs  ΔX_s（线性输出）
      - KL：对每个 stage 的 z_s 都做 KL，并按 beta 加权
    """
    assert reduction in ("mean", "sum")

    L_rec = 0.0
    L_kl = 0.0
    deltas = extras["deltas"]
    recons = extras["recons"]
    resolutions = extras["resolutions"]
    S = len(deltas)  # 已经包含 0..stage

    for s in range(S):
        res = resolutions[s]
        x_s = F.interpolate(x, size=(res, res), mode="area")

        if s == 0:
            # s=0：与基底图像直接做 BCE（MNIST 推荐）
            if dataset.lower() == "mnist":
                # deltas[0] 即 X'_0（Sigmoid 输出）
                bce = F.binary_cross_entropy(deltas[0], x_s, reduction=reduction)
                L_rec = L_rec + bce
            else:
                # 其他数据也可用 Gaussian（MSE）
                mse = F.mse_loss(deltas[0], x_s, reduction=reduction)
                L_rec = L_rec + mse
        else:
            # s>0：对残差做 Gaussian（MSE）
            prev_up = F.interpolate(recons[s - 1].detach(), size=(res, res), mode="nearest")
            D_s = x_s - prev_up
            mse = F.mse_loss(deltas[s], D_s, reduction=reduction)
            L_rec = L_rec + mse

        # KL 累加
        mu, logvar = mu_list[s], logvar_list[s]
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) if reduction == "mean" \
              else -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        L_kl = L_kl + kld

    total = L_rec + beta * L_kl
    stats = {
        "total": total.item() if torch.is_tensor(total) else total,
        "rec": L_rec.item() if torch.is_tensor(L_rec) else L_rec,
        "kl": L_kl.item() if torch.is_tensor(L_kl) else L_kl,
    }
    return total, stats


# -----------------------------
# 简单自测（shape 测试）
# -----------------------------
if __name__ == "__main__":
    B = 4
    x = torch.randn(B, 1, 32, 32)  # 假设已对齐到最终分辨率，内部会重采样到各 stage
    model = ResidualProgressiveVAE(in_ch=1, out_ch=1, start_res=4, final_res=32,
                                   latent_dim=64, base_ch=64, min_ch=16)

    for stage in range(model.max_stage + 1):
        alpha = 0.5  # 过渡期示意
        recon, mus, logvars, extras = model(x, stage=stage, alpha=alpha)
        print(f"stage={stage}  recon:{tuple(recon.shape)}  "
              f"stages_mu={len(mus)}  deltas_0_shape={tuple(extras['deltas'][0].shape)}")
        loss, stats = residual_pvae_loss(x, recon, mus, logvars, extras, dataset="mnist", beta=1.0)
        print("loss stats:", stats)