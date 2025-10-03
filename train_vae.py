import os
import argparse
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.datasets import MNIST

from utils import Logger, setup_experiment, compute_fid_for_vae
from models.VAE import VAE
from models.CNN_VAE import CNN_VAE
from loss import VAELoss


class FreyFaces(Dataset):
    def __init__(self, root: str, train: bool = True):
        super().__init__()
        
        # 加载.mat文件
        mat_path = os.path.join(root, "frey_rawface.mat")
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"未找到数据文件: {mat_path}")
        
        try:
            from scipy.io import loadmat
        except ImportError:
            raise RuntimeError("需要安装scipy来读取.mat文件: pip install scipy")
        
        # 加载并处理数据
        mat = loadmat(mat_path)
        arr = mat["ff"].T.reshape(-1, 28, 20).astype(np.float32)
        
        # 归一化到[0,1]
        if arr.max() > 1.0:
            arr = arr / 255.0
        
        # 划分训练/测试集 (9:1)
        split_idx = int(0.9 * len(arr))
        self.data = arr[:split_idx] if train else arr[split_idx:]
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.to_tensor(self.data[idx])
        return img, 0


def create_data_loaders(args) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
    """创建数据加载器"""
    if args.dataset == 'mnist':
        H, W = 28, 28
        transform = T.ToTensor()
        train_data = MNIST(args.data_path, transform=transform, train=True, download=True)
        test_data = MNIST(args.data_path, transform=transform, train=False, download=True)
    else:  # freyfaces
        H, W = 28, 20
        train_data = FreyFaces(args.data_path, train=True)
        test_data = FreyFaces(args.data_path, train=False)

    # DataLoader配置
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    
    return train_loader, test_loader, (H, W)


def create_model(args, H: int, W: int, device: torch.device) -> Tuple[torch.nn.Module, int]:
    """创建模型"""
    if args.model == "mlp":
        input_dim = args.input_dim or H * W
        model = VAE(input_dim, args.hidden_dim, args.z_dim)
    else:  # cnn
        if args.dataset == "freyfaces":
            raise NotImplementedError("CNN模型还未适配28x20尺寸")
        model = CNN_VAE(args.z_dim)
        input_dim = H * W
    
    return model.to(device), input_dim


def train_epoch(model, train_loader, loss_obj, optimizer, device, args, input_dim: int) -> Tuple[float, float, float]:
    """训练一个epoch"""
    model.train()
    total_loss = recon_loss = kld_loss = 0.0
    n_samples = 0

    for x, _ in train_loader:
        # 数据预处理
        if args.model == "mlp":
            x = x.view(-1, input_dim)
        x = x.to(device)

        # 前向传播
        optimizer.zero_grad()
        x_recon, mu, logvar, z = model(x)
        total, rec, kld = loss_obj(x, x_recon, mu, logvar, z=z)

        # 反向传播
        total.backward()
        optimizer.step()

        # 累计损失
        batch_size = x.size(0)
        n_samples += batch_size
        total_loss += total.item()
        recon_loss += rec.item()
        kld_loss += kld.item()

    return total_loss / n_samples, recon_loss / n_samples, kld_loss / n_samples


def evaluate_and_save(model, test_loader, device, args, exp_dir: str, H: int, W: int, input_dim: int):
    """评估模型并保存结果"""
    model.eval()
    with torch.no_grad():
        # 生成样本
        z = torch.randn(64, args.z_dim, device=device)
        samples = model.decode(z)
        if args.model == "mlp":
            samples = samples.view(64, 1, H, W)
        save_image(samples, os.path.join(exp_dir, 'samples.png'), nrow=8)

        # 重建测试
        test_x, _ = next(iter(test_loader))
        test_x = test_x[:32]
        if args.model == "mlp":
            test_x_input = test_x.view(-1, input_dim).to(device)
        else:
            test_x_input = test_x.to(device)

        recon_x, _, _, _ = model(test_x_input)
        
        # 创建对比图像
        if args.model == "mlp":
            original = test_x.view(-1, 1, H, W).to(device)
            reconstructed = recon_x.view(-1, 1, H, W)
        else:
            original = test_x.to(device)
            reconstructed = recon_x

        # 添加分隔线并保存
        separator = torch.zeros(8, 1, H, W, device=device)
        comparison = torch.cat([original, separator, reconstructed])
        save_image(comparison, os.path.join(exp_dir, 'reconstruction.png'), nrow=8)

        # 计算FID
        fid = compute_fid_for_vae(model, test_loader, device, H * W, args.z_dim, image_shape=(H, W))
        print(f"FID: {fid:.4f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VAE Training')
    
    # 基本参数
    parser.add_argument('--exp_name', type=str, default='vae', help='实验名称')
    parser.add_argument('--data_path', type=str, default='./data', help='数据路径')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'freyfaces'])
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'])
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, help='输入维度(仅MLP)，默认自动设置')
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--z_dim', type=int, default=20)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)

    # 损失函数参数
    parser.add_argument('--recon', type=str, default='bce', choices=['bce', 'gaussian'])
    parser.add_argument('--kl', type=str, default='analytic', choices=['analytic', 'mc'])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--reduction', type=str, default='sum', choices=['sum', 'mean'])
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian重建损失的σ')
    parser.add_argument('--sigma_learnable', action='store_true', help='使σ可学习')
    parser.add_argument('--include_const', action='store_true', help='Gaussian NLL包含常数项')

    args = parser.parse_args()
    args.exp_name = f"{args.model}_{args.dataset}_{args.recon}_{args.kl}_{args.exp_name}"
    return args


def main(args):
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = setup_experiment(args)

    with Logger(os.path.join(exp_dir, 'training.log')):
        # 数据加载
        train_loader, test_loader, (H, W) = create_data_loaders(args)
        
        # 模型创建
        model, input_dim = create_model(args, H, W, device)
        
        # 损失函数
        loss_obj = VAELoss(
            recon_type=args.recon,
            kl_type=args.kl,
            sigma=args.sigma,
            learnable_sigma=args.sigma_learnable,
            include_const=args.include_const,
            beta=args.beta,
            reduction=args.reduction,
        ).to(device)

        # 优化器
        params = list(model.parameters())
        if args.sigma_learnable:
            params.extend(loss_obj.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)

        # 训练循环
        print(f"开始训练 {args.model.upper()}_VAE，数据集: {args.dataset}")
        for epoch in range(args.epochs):
            total_loss, recon_loss, kld_loss = train_epoch(
                model, train_loader, loss_obj, optimizer, device, args, input_dim
            )
            
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Loss: {total_loss:.4f}, Recon: {recon_loss:.4f}, KLD: {kld_loss:.4f}")

        print("训练完成!")

        # 保存模型
        torch.save(model.state_dict(), os.path.join(exp_dir, f"{args.model}_vae.pth"))

        # 评估和可视化
        evaluate_and_save(model, test_loader, device, args, exp_dir, H, W, input_dim)
        print(f"结果已保存到 {exp_dir}")


if __name__ == '__main__':
    main(parse_args())