import os
import sys
import json
from datetime import datetime
import torch
from torchvision import models


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

    def __enter__(self):
        self._prev_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = getattr(self, '_prev_stdout', self.terminal)
        self.close()

def setup_experiment(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("experiments", f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    config = vars(args).copy()
    config['device'] = str(args.device)
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"实验目录: {exp_dir}")
    return exp_dir


# ---------------------------
# FID 计算相关工具函数
# ---------------------------

def _preprocess_for_inception(x: torch.Tensor) -> torch.Tensor:
    """将输入张量规范到 InceptionV3 所需的形状与分布。
    输入 x: (B, C=1, H, W) 且像素范围 [0,1]
    输出: (B, 3, 299, 299) 正则化后张量
    """
    if x.dtype != torch.float32:
        x = x.float()
    x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    x = x.repeat(1, 3, 1, 1)  # 灰度转为 3 通道
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def _get_inception_model(device: torch.device):
    """加载 InceptionV3 模型并返回模型与一个提取 pool3 特征的函数。"""
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()

    def get_features(imgs: torch.Tensor) -> torch.Tensor:
        feats = []
        def hook(_m, _i, o):
            feats.append(o)
        h = model.avgpool.register_forward_hook(hook)
        with torch.no_grad():
            _ = model(imgs)
        h.remove()
        f = feats[0]
        return torch.flatten(f, start_dim=1)

    return model, get_features


def _stats(feats: torch.Tensor):
    """计算均值与协方差（无偏）。feats: (N, D)"""
    mu = feats.mean(dim=0)
    x = feats - mu
    n = feats.shape[0]
    cov = (x.T @ x) / (n - 1)
    return mu, cov


def _symmetric_matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """对称(半正定)矩阵的平方根，使用特征分解实现。"""
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0)
    sqrt_vals = torch.sqrt(eigvals)
    return (eigvecs @ torch.diag(sqrt_vals) @ eigvecs.T)


def _frechet_distance(mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor) -> float:
    """计算 Frechet 距离。"""
    cov1_sqrt = _symmetric_matrix_sqrt(cov1)
    mid = cov1_sqrt @ cov2 @ cov1_sqrt
    mid_sqrt = _symmetric_matrix_sqrt(mid)
    diff = (mu1 - mu2)
    fid = diff.dot(diff).item() + torch.trace(cov1).item() + torch.trace(cov2).item() - 2.0 * torch.trace(mid_sqrt).item()
    return float(fid)


def compute_fid_for_vae(vae_model, test_loader, device: torch.device, input_dim: int, latent_dim: int, n_batches: int | None = None) -> float:
    """对 VAE 的生成分布与真实测试集计算 FID。
    - vae_model: 训练好的 VAE（需含 decoder）
    - test_loader: 测试数据 DataLoader，提供真实图像 (B,1,H,W)
    - device: 设备
    - input_dim: 输入维度（MNIST=784）
    - latent_dim: 潜在维度
    - n_batches: 限制参与计算的批次数（None 则使用全部）
    """
    inception, get_feats = _get_inception_model(device)
    inception.eval()

    real_feats_list = []
    fake_feats_list = []
    side = int(input_dim ** 0.5)

    with torch.no_grad():
        for b_idx, (x_real, _) in enumerate(test_loader):
            if n_batches is not None and b_idx >= n_batches:
                break
            # 真实图像特征
            x_r = _preprocess_for_inception(x_real.to(device))
            fr = get_feats(x_r)
            real_feats_list.append(fr.cpu())

            # 生成图像特征（与真实批大小一致）
            z = torch.randn(x_real.size(0), latent_dim, device=device)
            x_fake_flat = vae_model.decoder(z)
            x_fake = x_fake_flat.view(-1, 1, side, side)
            x_f = _preprocess_for_inception(x_fake)
            ff = get_feats(x_f)
            fake_feats_list.append(ff.cpu())

    real_feats = torch.cat(real_feats_list, dim=0)
    fake_feats = torch.cat(fake_feats_list, dim=0)
    mu_r, cov_r = _stats(real_feats)
    mu_f, cov_f = _stats(fake_feats)
    return _frechet_distance(mu_r, cov_r, mu_f, cov_f)