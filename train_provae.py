# provae_demo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import argparse
from typing import Tuple

# --- 1. 参数配置 (Configuration) ---
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Progressive VAE (ProVAE).")
    parser.add_argument('--exp_name', type=str, default='provae_mnist_demo', help="Name for the experiment output folder.")
    parser.add_argument('--data_path', type=str, default='./data', help="Path to MNIST dataset.")
    parser.add_argument('--output_dir', type=str, default='./experiments', help="Directory to save results.")
    parser.add_argument('--z_dim', type=int, default=16, help="Dimension of the latent space z.")
    parser.add_argument('--base_ch', type=int, default=64, help="Base channel count for convolutional layers.")
    parser.add_argument('--max_stage', type=int, default=3, help="Maximum stage (0=4x4, 1=8x8, 2=16x16, 3=32x32).")
    parser.add_argument('--epochs_per_stage', type=int, default=5, help="Number of epochs to train for each stage.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the Adam optimizer.")
    
    args = parser.parse_args()
    return args

# --- 2. 模型构建块 (Model Building Blocks) ---
class ConvBlock(nn.Module):
    """一个包含两个卷积层和LeakyReLU激活函数的基础块"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- 3. 渐进式模型定义 (Progressive Model Definition) ---
class ProgressiveEncoder(nn.Module):
    """
    ProVAE的编码器部分，可以逐步处理更高分辨率的输入。
    """
    def __init__(self, in_ch: int, z_dim: int, base_ch: int, max_stage: int):
        super().__init__()
        self.max_stage = max_stage
        
        # 每个stage的特征通道数
        channels = [base_ch] * (max_stage + 1)
        
        # from_rgb: 将输入图像转换为特征图
        self.from_rgb = nn.ModuleList([nn.Conv2d(in_ch, channels[s], kernel_size=1) for s in range(max_stage + 1)])
        
        # blocks: 存储所有下采样块
        self.blocks = nn.ModuleList()
        for s in range(max_stage, 0, -1):
            self.blocks.append(nn.Sequential(ConvBlock(channels[s], channels[s-1]), nn.AvgPool2d(2)))

        self.final_conv = ConvBlock(channels[0], channels[0])
        self.mu = nn.Linear(channels[0] * 4 * 4, z_dim)
        self.logvar = nn.Linear(channels[0] * 4 * 4, z_dim)

    def forward(self, x: torch.Tensor, stage: int, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if stage == 0:
            h = self.from_rgb[0](x)
        else:
            # 新路径 (当前分辨率)
            h_new = self.from_rgb[stage](x)
            
            # 旧路径 (先下采样一半，再走上一个stage的from_rgb)
            x_down = F.avg_pool2d(x, 2)
            h_old = self.from_rgb[stage - 1](x_down)
            
            # 混合两个路径 - 需要让h_new下采样到与h_old相同的分辨率
            h_new_down = F.avg_pool2d(h_new, 2)
            h = (1 - alpha) * h_old + alpha * h_new_down
        
        # 从混合后的特征继续下采样到4x4
        # 计算当前h对应的stage (对于stage > 0，h现在相当于stage-1的分辨率)
        current_stage = stage - 1 if stage > 0 else 0
        
        # 从当前分辨率下采样到4x4
        for s in range(current_stage, 0, -1):
            # blocks[0]对应从max_stage到max_stage-1的下采样
            # blocks[i]对应从max_stage-i到max_stage-i-1的下采样
            block_idx = self.max_stage - s
            h = self.blocks[block_idx](h)
            
        h = self.final_conv(h)
        h = h.view(h.size(0), -1)
        return self.mu(h), self.logvar(h)

class ProgressiveDecoder(nn.Module):
    """
    ProVAE的解码器部分，可以逐步生成更高分辨率的输出。
    """
    def __init__(self, z_dim: int, out_ch: int, base_ch: int, max_stage: int):
        super().__init__()
        self.max_stage = max_stage
        channels = [base_ch] * (max_stage + 1)
        
        self.fc = nn.Linear(z_dim, channels[0] * 4 * 4)
        
        # blocks: 存储所有上采样块
        self.blocks = nn.ModuleList()
        for s in range(max_stage):
            self.blocks.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), ConvBlock(channels[s], channels[s+1])))
        
        # to_rgb: 将特征图转换为图像
        self.to_rgb = nn.ModuleList([nn.Conv2d(channels[s], out_ch, kernel_size=1) for s in range(max_stage + 1)])

    def forward(self, z: torch.Tensor, stage: int, alpha: float) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        
        if stage == 0:
            return torch.sigmoid(self.to_rgb[0](h))

        # 逐步上采样到上一个stage
        for s in range(int(stage) - 1):
            h = self.blocks[s](h)
        
        # 旧路径 (上一个分辨率输出)
        rgb_old = self.to_rgb[stage-1](h)
        rgb_old_upsampled = F.interpolate(rgb_old, scale_factor=2, mode='nearest')
        
        # 新路径 (当前分辨率输出)
        h_new = self.blocks[int(stage) - 1](h)
        rgb_new = self.to_rgb[int(stage)](h_new)
        
        # Fade-in混合输出，应用sigmoid
        return torch.sigmoid((1 - alpha) * rgb_old_upsampled + alpha * rgb_new)

class ProVAE(nn.Module):
    """ProVAE模型，整合了编码器和解码器"""
    def __init__(self, in_ch: int, z_dim: int, base_ch: int, max_stage: int):
        super().__init__()
        self.encoder = ProgressiveEncoder(in_ch, z_dim, base_ch, max_stage)
        self.decoder = ProgressiveDecoder(z_dim, in_ch, base_ch, max_stage)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, stage: int, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, stage, alpha)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, stage, alpha)
        return x_recon, mu, logvar

# --- 4. 损失函数 (Loss Function) ---
def vae_loss_function(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE损失函数，与VAE模型保持一致的命名和形式"""
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

# --- 5. 主训练脚本 (Main Training Script) ---
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 数据加载 (Data Loading)
    transform = transforms.Compose([
        transforms.Pad(2),  # MNIST 28x28 -> 32x32，方便下采样
        transforms.ToTensor(),
    ])
    train_data = MNIST(args.data_path, transform=transform, train=True, download=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 模型与优化器 (Model and Optimizer)
    model = ProVAE(in_ch=1, z_dim=args.z_dim, base_ch=args.base_ch, max_stage=args.max_stage).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"开始训练 ProVAE on device={device}...")
    print(f"实验结果将保存在: {exp_dir}")

    # 渐进式训练循环 (Progressive Training Loop)
    for stage in range(args.max_stage + 1):
        res = 4 * (2 ** stage)
        print(f"\n--- Stage {stage}: Resolution {res}x{res} ---")
        
        for epoch in range(args.epochs_per_stage):
            # alpha从0线性增长到1，实现fade-in
            alpha = (epoch + 1) / args.epochs_per_stage
            
            model.train()
            total_loss_sum = recon_loss_sum = kl_div_sum = 0.0
            
            for x, _ in train_loader:
                # 调整输入图像到当前stage的分辨率
                x = F.interpolate(x, size=(res, res), mode='nearest').to(device)
                
                optimizer.zero_grad()
                recon_x, mu, logvar = model(x, stage, alpha)
                loss, BCE, KLD = vae_loss_function(recon_x, x, mu, logvar)
                
                loss.backward()
                optimizer.step()
                
                total_loss_sum += loss.item()
                recon_loss_sum += BCE.item()
                kl_div_sum += KLD.item()

            # 打印每个epoch的平均损失
            num_batches = len(train_loader)
            print(f"  Epoch {epoch+1:02d}/{args.epochs_per_stage} | Alpha: {alpha:.2f} | "
                  f"Loss: {total_loss_sum/num_batches:.4f} (Recon: {recon_loss_sum/num_batches:.4f}, KL: {kl_div_sum/num_batches:.4f})")

        # 每个stage结束时，保存生成和重建的样本
        model.eval()
        with torch.no_grad():
            # 生成随机样本
            z_sample = torch.randn(64, args.z_dim).to(device)
            generated = torch.sigmoid(model.decoder(z_sample, stage, alpha=1.0))
            save_image(generated, os.path.join(exp_dir, f'stage_{stage}_generated.png'), nrow=8)
            
            # 重建样本
            x_test, _ = next(iter(train_loader))
            x_test = F.interpolate(x_test, size=(res, res), mode='nearest').to(device)
            recon_logits, _, _ = model(x_test, stage, alpha=1.0)
            recon_images = torch.sigmoid(recon_logits)
            
            # 将原图和重建图拼接在一起进行对比
            comparison = torch.cat([x_test[:8], recon_images[:8]])
            save_image(comparison, os.path.join(exp_dir, f'stage_{stage}_reconstruction.png'), nrow=8)
            
        print(f"  Stage {stage} sample images saved.")
    
    print("\n训练完成!")
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(exp_dir, 'provae_final.pth'))
    print(f"Final model saved to {os.path.join(exp_dir, 'provae_final.pth')}")

if __name__ == '__main__':
    main()