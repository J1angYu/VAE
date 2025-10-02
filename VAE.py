import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
import argparse
import json
from utils import Logger, setup_experiment, compute_fid_for_vae
from models import VAE, vae_loss


def parse_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Training')
    parser.add_argument('--exp_name', type=str, default='vae_mnist', help='实验名称')
    parser.add_argument('--data_path', type=str, default='./data', help='数据路径')
    parser.add_argument('--input_dim', type=int, default=784, help='输入维度')
    parser.add_argument('--hidden_dim', type=int, default=400, help='隐藏维度')
    parser.add_argument('--latent_dim', type=int, default=2, help='潜在维度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--device', type=str, default='auto', help='设备: auto/cpu/cuda')
    parser.add_argument('--fid_batches', type=int, default=-1, help='FID评估批次数（-1为全部）')
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == 'auto' else torch.device(args.device)
    return args


def main(args):
    exp_dir = setup_experiment(args)
    
    with Logger(os.path.join(exp_dir, 'training.log')) as logger:
        # 数据加载
        os.makedirs(args.data_path, exist_ok=True)
        transform = transforms.ToTensor()
        train_data = MNIST(args.data_path, transform=transform, train=True, download=True)
        test_data = MNIST(args.data_path, transform=transform, train=False, download=True)
        
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.device.type == "cuda" else {}
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

        # 模型初始化
        model = VAE(args.input_dim, args.hidden_dim, args.latent_dim).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 训练
        print("Start training...")
        model.train()
        for epoch in range(args.epochs):
            total_loss = recon_loss_sum = kl_loss_sum = 0
            
            for x, _ in train_loader:
                x = x.view(-1, args.input_dim).to(args.device)
                optimizer.zero_grad()
                
                x_recon, mu, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar)
                
                total_loss += loss.item()
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_loss.item()
                
                loss.backward()
                optimizer.step()
            
            n_samples = len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Loss: {total_loss/n_samples:.4f}, "
                  f"Recon: {recon_loss_sum/n_samples:.4f}, "
                  f"KL: {kl_loss_sum/n_samples:.4f}")

        print("Finish training!")

        # 保存模型
        torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
        
        # 生成和重建图像
        model.eval()
        with torch.no_grad():
            # 生成图像
            z = torch.randn(args.batch_size, args.latent_dim).to(args.device)
            generated = model.decoder(z).view(args.batch_size, 1, 28, 28)
            save_image(generated, os.path.join(exp_dir, 'generated.png'))
            
            # 重建图像
            test_x, _ = next(iter(test_loader))
            test_x = test_x.view(-1, args.input_dim).to(args.device)
            recon_x, _, _ = model(test_x)
            
            comparison = torch.cat([
                test_x.view(args.batch_size, 1, 28, 28),
                recon_x.view(args.batch_size, 1, 28, 28)
            ])
            save_image(comparison, os.path.join(exp_dir, 'reconstruction.png'), 
                      nrow=int(args.batch_size**0.5))
            
            print(f"结果已保存到 {exp_dir}")

            # 计算并保存 FID 指标
            n_batches = None if args.fid_batches == -1 else args.fid_batches
            fid = compute_fid_for_vae(model, test_loader, args.device, args.input_dim, args.latent_dim, n_batches=n_batches)
            print(f"FID: {fid:.4f}")
            with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
                json.dump({'FID': fid}, f, indent=2)
    
if __name__ == '__main__':
    main(parse_args())