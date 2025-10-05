import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def recon_bce(x, x_recon, reduction="sum"):
    """BCE重建损失，要求x_recon已经过sigmoid"""
    return F.binary_cross_entropy(x_recon, x, reduction=reduction)


def recon_gaussian(x, x_recon, sigma=1.0, log_sigma_param=None, 
                   include_const=False, reduction="sum"):
    """高斯重建损失：0.5 * ||x - x̂||^2 / σ^2 + [常数项]"""
    # 计算方差
    if log_sigma_param is not None:
        sigma2 = torch.exp(2.0 * log_sigma_param)
    else:
        sigma2 = torch.tensor(sigma ** 2, device=x.device, dtype=x.dtype)

    # 主要损失项
    err2 = (x - x_recon) ** 2
    nll_quad = 0.5 * (torch.sum(err2 / sigma2) if reduction == "sum" 
                      else torch.mean(err2 / sigma2))

    # 可选常数项
    if include_const:
        numel = x.numel() if reduction == "sum" else 1.0
        const = 0.5 * (math.log(2.0 * math.pi) + torch.log(sigma2)) * numel
        return nll_quad + const
    
    return nll_quad


def kl_analytic(mu, logvar, reduction="sum"):
    """解析KL散度：KL(q(z|x)||p(z))，p(z)=N(0,I)"""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.sum(kl) if reduction == "sum" else torch.mean(kl)


def kl_per_sample(mu, logvar, z=None, reduction="sum"):
    """蒙特卡洛KL散度估计"""
    if z is None:
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

    # 计算对数密度
    k = mu.size(1)
    log_q = -0.5 * torch.sum(
        math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp(), 
        dim=1
    )
    log_p = -0.5 * torch.sum(math.log(2 * math.pi) + z.pow(2), dim=1)
    kl_each = log_q - log_p

    return torch.sum(kl_each) if reduction == "sum" else torch.mean(kl_each)


class VAELoss(nn.Module):
    """VAE损失函数，支持多种重建损失和KL散度计算方式"""
    
    def __init__(self, recon_type="bce", kl_type="analytic", sigma=1.0,
                 learnable_sigma=False, include_const=False, beta=1.0, 
                 reduction="sum"):
        super().__init__()
        
        # 参数验证
        assert recon_type in ("bce", "gaussian"), f"不支持的重建损失类型: {recon_type}"
        assert kl_type in ("analytic", "mc"), f"不支持的KL类型: {kl_type}"
        assert reduction in ("sum", "mean"), f"不支持的reduction: {reduction}"

        self.recon_type = recon_type
        self.kl_type = kl_type
        self.beta = beta
        self.reduction = reduction
        self.include_const = include_const
        self.fixed_sigma = sigma

        # 可学习的sigma
        if recon_type == "gaussian" and learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        else:
            self.log_sigma = None

    def forward(self, x, x_recon, mu, logvar, z=None):
        """计算VAE总损失"""
        # 重建损失
        if self.recon_type == "bce":
            rec_loss = recon_bce(x, x_recon, self.reduction)
        else:
            rec_loss = recon_gaussian(
                x, x_recon, self.fixed_sigma, self.log_sigma,
                self.include_const, self.reduction
            )

        # KL散度
        if self.kl_type == "analytic":
            kl_loss = kl_analytic(mu, logvar, self.reduction)
        else:
            kl_loss = kl_per_sample(mu, logvar, z, self.reduction)

        total_loss = rec_loss + self.beta * kl_loss
        return total_loss, rec_loss, kl_loss