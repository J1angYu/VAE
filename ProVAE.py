import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
import os
import json
from datetime import datetime

# --- 1. 配置 ---
CONFIG = {
    "experiment_name": "pro_vae_mnist",
    "dataset_path": './data',
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # 模型参数
    "latent_dim": 32,
    "image_channels": 1,
    "base_channels": 256,
    "max_channels": 512,
    
    # 训练参数
    "lr": 1e-3,
    "resolutions": [4, 8, 16, 32],
    "batch_sizes": {4: 256, 8: 256, 16: 128, 32: 64},
    "epochs_per_stage": {4: 2, 8: 2, 16: 2, 32: 2},
    "fade_in_percentage": 0.5,
}

# --- 2. 辅助函数：设置实验环境 ---
def setup_experiment(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = f"{config['experiment_name']}_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device'])
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config_to_save, f, indent=4)
        
    print(f"Experiment setup complete. Results will be saved in: {exp_dir}")
    return exp_dir

# --- 3. 模型定义 ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.lrelu(self.conv(x))

class ProgressiveVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['latent_dim']
        self.img_channels = config['image_channels']
        base_c = config['base_channels']

        # 核心潜变量层
        self.fc_mu = nn.Linear(base_c * 2 * 2, self.latent_dim)
        self.fc_log_var = nn.Linear(base_c * 2 * 2, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, base_c * 4 * 4)

        # 初始4x4块
        self.initial_decoder = ConvBlock(base_c, base_c)
        self.initial_encoder = ConvBlock(base_c, base_c)
        
        # 动态增长的块
        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        
        self.from_rgbs = nn.ModuleList([ConvBlock(self.img_channels, base_c, 1, 1, 0)])
        self.to_rgbs = nn.ModuleList([nn.Conv2d(base_c, self.img_channels, 1, 1, 0)])
        
        self.downscale = nn.AvgPool2d(2)
        self.upscale = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def grow(self):
        # 获取当前最高分辨率层的通道数
        current_channels = self.from_rgbs[-1].conv.out_channels
        
        # 对于编码器: 保持通道数不变(类似ProGAN的判别器在中等分辨率)
        # 对于解码器: 保持通道数不变
        new_channels = current_channels
        
        # 添加新的编码器和解码器块
        self.encoder_blocks.append(ConvBlock(new_channels, new_channels))
        self.decoder_blocks.append(ConvBlock(new_channels, new_channels))
        
        # 添加新的from_rgb和to_rgb层
        self.from_rgbs.append(ConvBlock(self.img_channels, new_channels, 1, 1, 0))
        self.to_rgbs.append(nn.Conv2d(new_channels, self.img_channels, 1, 1, 0))

    def forward(self, x, alpha, stage):
        # --- 编码器 ---
        h = self.from_rgbs[stage](x)
        
        # 从当前stage开始，逐步下采样
        for i in range(stage, 0, -1):
            if i == stage and stage > 0 and alpha < 1.0:
                # Fade-in逻辑
                h = self.downscale(h)
                prev_h = self.from_rgbs[stage-1](self.downscale(x))
                h = (1 - alpha) * prev_h + alpha * h
            else:
                h = self.downscale(h)
            
            h = self.encoder_blocks[i-1](h)
            
        h = self.initial_encoder(self.downscale(h))
        
        h = h.view(h.size(0), -1)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        
        # --- 解码器 ---
        h = self.fc_decode(z).view(z.size(0), -1, 4, 4)
        h = self.initial_decoder(h)

        for i in range(stage):
            h = self.upscale(h)
            h = self.decoder_blocks[i](h)

        if stage > 0 and alpha < 1.0:
            # 解码器的fade-in逻辑
            prev_out = self.to_rgbs[stage-1](h)
            prev_out = self.upscale(prev_out)
            
            h = self.upscale(h)
            h = self.decoder_blocks[stage-1](h)
            curr_out = self.to_rgbs[stage](h)
            
            x_hat = (1 - alpha) * prev_out + alpha * curr_out
        else:
            x_hat = self.to_rgbs[stage](h)
            
        return torch.sigmoid(x_hat), mu, log_var

# --- 4. 损失函数 ---
def loss_function(x, x_hat, mu, log_var):
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld

# --- 5. 核心功能函数 ---
def train_stage(model, loader, optimizer, device, stage, epochs, fade_in_epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        alpha = min(1.0, (epoch + 1) / fade_in_epochs) if stage > 0 else 1.0

        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(x, alpha, stage)
            loss = loss_function(x, x_hat, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Res: {x.shape[-1]}x{x.shape[-1]}, Alpha: {alpha:.2f}, Loss: {avg_loss:.4f}")

def evaluate_and_save(model, device, latent_dim, stage, res, exp_dir):
    model.eval()
    with torch.no_grad():
        # 重建
        transform = transforms.Compose([transforms.Resize(res), transforms.ToTensor()])
        test_dataset = MNIST(CONFIG["dataset_path"], train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        real_images = next(iter(test_loader))[0].to(device)
        recon_images, _, _ = model(real_images, alpha=1.0, stage=stage)
        comparison = torch.cat([real_images, recon_images])
        save_image(comparison.cpu(), os.path.join(exp_dir, f"reconstruction_stage_{stage}_{res}x{res}.png"), nrow=8)
        
        # 生成
        noise = torch.randn(64, latent_dim).to(device)
        h = model.fc_decode(noise).view(-1, CONFIG["base_channels"], 4, 4)
        h = model.initial_decoder(h)
        for i in range(stage):
            h = model.upscale(h)
            h = model.decoder_blocks[i](h)
        generated = torch.sigmoid(model.to_rgbs[stage](h))
        save_image(generated.cpu(), os.path.join(exp_dir, f"generated_stage_{stage}_{res}x{res}.png"))
    print(f"Saved images for stage {stage} to {exp_dir}")

# --- 6. 主执行函数 ---
def main(config):
    exp_dir = setup_experiment(config)
    device = config['device']
    
    model = ProgressiveVAE(config).to(device)
    
    for stage, res in enumerate(config['resolutions']):
        print(f"\n--- Training Stage {stage}: Resolution {res}x{res} ---")
        
        if stage > 0:
            model.grow()
            model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        transform = transforms.Compose([transforms.Resize(res), transforms.ToTensor()])
        dataset = MNIST(config["dataset_path"], train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=config['batch_sizes'][res], shuffle=True, num_workers=4, pin_memory=True)
        
        epochs = config['epochs_per_stage'][res]
        fade_in_epochs = int(epochs * config['fade_in_percentage'])
        
        train_stage(model, loader, optimizer, device, stage, epochs, fade_in_epochs)
        evaluate_and_save(model, device, config['latent_dim'], stage, res, exp_dir)

    print("\n--- Training Finished! ---")
    final_model_path = os.path.join(exp_dir, "pro_vae_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    main(CONFIG)