import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
import json
from datetime import datetime

# --- 1. 配置 ---
CONFIG = {
    "experiment_name": "vae_mnist",
    "dataset_path": './data',
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "x_dim": 784,
    "hidden_dim": 400,
    "latent_dim": 32,
    "batch_size": 128,
    "lr": 1e-3,
    "epochs": 30,
}

# --- 辅助函数：设置实验环境 ---
def setup_experiment(config):
    """
    根据配置创建实验目录，并保存配置文件。
    """
    # 1. 创建基于时间戳的唯一实验目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = f"{config['experiment_name']}_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 2. 保存配置文件
    config_path = os.path.join(exp_dir, "config.json")
    
    # 准备一个可序列化的config副本
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device'])
    
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
        
    print(f"Experiment setup complete. Results will be saved in: {exp_dir}")
    return exp_dir

# --- 3. 模型定义 ---
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

# --- 4. 损失函数 ---
def loss_function(x, x_hat, mu, log_var):
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld

# --- 主执行函数 ---
def main(config):
    # 1. 设置实验环境
    exp_dir = setup_experiment(config)
    device = config['device']
    
    # 2. 数据加载
    os.makedirs(config["dataset_path"], exist_ok=True)
    mnist_transform = transforms.ToTensor()
    train_dataset = MNIST(config["dataset_path"], transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(config["dataset_path"], transform=mnist_transform, train=False, download=True)
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=config["batch_size"], shuffle=False, **kwargs)

    # 3. 初始化模型和优化器
    model = VAE(
        input_dim=config["x_dim"], 
        hidden_dim=config["hidden_dim"], 
        latent_dim=config["latent_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # 4. 训练循环
    print("Start training VAE...")
    model.train()
    for epoch in range(config["epochs"]):
        overall_loss = 0
        for x, _ in train_loader:
            x = x.view(-1, config["x_dim"]).to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(x)
            loss = loss_function(x, x_hat, mu, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = overall_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{config['epochs']} complete! \tAverage Loss: {avg_loss:.4f}")

    print("Finish training!!")

    # 5. 保存模型
    model_path = os.path.join(exp_dir, 'model_final.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 6. 生成、可视化并保存结果
    model.eval()
    with torch.no_grad():
        # 从噪声生成图像
        noise = torch.randn(config["batch_size"], config["latent_dim"]).to(device)
        generated_images = model.decoder(noise) 
        generated_img_path = os.path.join(exp_dir, 'generated_sample.png')
        save_image(generated_images.view(config["batch_size"], 1, 28, 28), generated_img_path)
        print(f"Saved generated samples to {generated_img_path}")

        # 从测试集重建图像
        test_images, _ = next(iter(test_loader))
        test_images = test_images.view(-1, config["x_dim"]).to(device)
        recons_images, _, _ = model(test_images)
        
        comparison = torch.cat([
            test_images.view(config["batch_size"], 1, 28, 28), 
            recons_images.view(config["batch_size"], 1, 28, 28)
        ])
        recons_img_path = os.path.join(exp_dir, 'reconstruction.png')
        save_image(comparison.cpu(), recons_img_path, nrow=int(config["batch_size"]**0.5))
        print(f"Saved reconstruction comparison to {recons_img_path}")

if __name__ == '__main__':
    main(CONFIG)