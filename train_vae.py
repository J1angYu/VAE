import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
import argparse
from utils import Logger, setup_experiment, compute_fid_for_vae
from models.VAE import VAE, vae_loss
from models.CNN_VAE import CNN_VAE, vae_loss as cnn_vae_loss

def parse_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Training')
    parser.add_argument('--exp_name', type=str, default='vae_mnist', help='实验名称')
    parser.add_argument('--data_path', type=str, default='./data', help='数据路径')
    parser.add_argument('--input_dim', type=int, default=784, help='输入维度')
    parser.add_argument('--hidden_dim', type=int, default=400, help='隐藏维度')
    parser.add_argument('--z_dim', type=int, default=20, help='潜在维度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='模型类型')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = setup_experiment(args)

    with Logger(os.path.join(exp_dir, 'training.log')) as logger:
        # 数据加载
        os.makedirs(args.data_path, exist_ok=True)
        transform = transforms.ToTensor()
        train_data = MNIST(args.data_path, transform=transform, train=True, download=True)
        test_data = MNIST(args.data_path, transform=transform, train=False, download=True)

        kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

        # 模型选择
        if args.model == "mlp":
            model = VAE(args.input_dim, args.hidden_dim, args.z_dim).to(device)
            loss_fn = vae_loss
        else:
            model = CNN_VAE(args.z_dim).to(device)
            loss_fn = cnn_vae_loss

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 训练
        print(f"Start training {args.model.upper()}_VAE...")
        model.train()
        for epoch in range(args.epochs):
            total_loss = recon_loss_sum = kld_loss_sum = 0

            for x, _ in train_loader:
                if args.model == "mlp":
                    x = x.view(-1, args.input_dim).to(device)
                else:  # CNN_VAE
                    x = x.to(device)

                optimizer.zero_grad()
                x_recon, mu, logvar = model(x)
                loss, recon_loss, kld_loss = loss_fn(x, x_recon, mu, logvar)

                total_loss += loss.item()
                recon_loss_sum += recon_loss.item()
                kld_loss_sum += kld_loss.item()

                loss.backward()
                optimizer.step()

            n_samples = len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Loss: {total_loss/n_samples:.4f}, "
                  f"Recon: {recon_loss_sum/n_samples:.4f}, "
                  f"KLD: {kld_loss_sum/n_samples:.4f}")

        print("Finish training!")

        # 保存模型
        torch.save(model.state_dict(), os.path.join(exp_dir, f"{args.model}_vae.pth"))

        # 生成和重建图像
        model.eval()
        with torch.no_grad():
            # 生成图像
            z = torch.randn(64, args.z_dim).to(device)
            sample = model.decode(z)
            save_image(sample.view(64, 1, 28, 28),
                       os.path.join(exp_dir, 'sample.png'), nrow=8)

            # 重建图像
            test_x, _ = next(iter(test_loader))
            if args.model == "mlp":
                test_x = test_x[:32].view(-1, args.input_dim).to(device)
            else:
                test_x = test_x[:32].to(device)

            recon_x, _, _ = model(test_x)
            comparison = torch.cat([
                test_x.view(-1, 1, 28, 28),  # 原始图
                torch.zeros(8, 1, 28, 28).to(device),   # 间隔
                recon_x.view(-1, 1, 28, 28)  # 重建图
            ])
            save_image(comparison, os.path.join(exp_dir, 'reconstruction.png'), nrow=8)
            print(f"结果已保存到 {exp_dir}")

            # 计算 FID
            fid = compute_fid_for_vae(model, test_loader, device,
                                      args.input_dim, args.z_dim)
            print(f"FID: {fid:.4f}")


if __name__ == '__main__':
    main(parse_args())