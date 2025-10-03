import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
import argparse
from utils import Logger, setup_experiment, compute_fid_for_vae
from models.ProVAE import ProVAE
from loss import vae_bce_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Progressive VAE MNIST Training')
    parser.add_argument('--exp_name', type=str, default='pro_vae_mnist', help='实验名称')
    parser.add_argument('--data_path', type=str, default='./data', help='数据路径')
    parser.add_argument('--z_dim', type=int, default=20, help='潜在维度')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    # progressive 相关
    parser.add_argument('--epochs_per_stage', type=int, default=10, help='每个stage训练的epoch数')
    parser.add_argument('--fadein_ratio', type=float, default=0.5, help='每stage中用于fade-in的比例(0~1)')
    parser.add_argument('--start_res', type=int, default=4, help='起始分辨率')
    parser.add_argument('--final_res', type=int, default=32, help='最终分辨率')
    parser.add_argument('--base_ch', type=int, default=128, help='最低分辨率通道数')
    parser.add_argument('--min_ch', type=int, default=16, help='最小通道数')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = setup_experiment(args)

    with Logger(os.path.join(exp_dir, 'training.log')) as logger:
        os.makedirs(args.data_path, exist_ok=True)
        transform = transforms.Compose([
            transforms.Pad(( (args.final_res - 28) // 2 )),
            transforms.ToTensor()
        ])
        train_data = MNIST(args.data_path, transform=transform, train=True, download=True)
        test_data = MNIST(args.data_path, transform=transform, train=False, download=True)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # ========= 模型 =========
        model = ProVAE(
            in_ch=1, z_dim=args.z_dim,
            start_res=args.start_res, final_res=args.final_res,
            base_ch=args.base_ch, min_ch=args.min_ch
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        print("Start training Progressive VAE...")

        global_step = 0
        for stage in range(model.max_stage + 1):
            cur_res = model._res_of(stage)
            n_epochs = args.epochs_per_stage
            n_fade = int(n_epochs * args.fadein_ratio) if stage > 0 else 0
            print(f"\n===> Stage {stage} (res={cur_res}x{cur_res}) | fade-in epochs: {n_fade}/{n_epochs}")

            for epoch in range(n_epochs):
                model.train()
                total_loss = recon_sum = kld_sum = 0.0

                if stage > 0 and n_fade > 0 and epoch < n_fade:
                    alpha_start = (epoch + 0.0) / n_fade
                    alpha_end   = (epoch + 1.0) / n_fade
                    alpha_start = max(0.0, min(1.0, alpha_start))
                    alpha_end   = max(0.0, min(1.0, alpha_end))
                    alpha_info = f"[alpha {alpha_start:.2f}->{alpha_end:.2f}]"
                else:
                    alpha_info = "[alpha 1.00]"

                for i, (x, _) in enumerate(train_loader):
                    x = x.to(device)
                    x_tgt = F.interpolate(x, size=(cur_res, cur_res), mode="area")

                    # per-step alpha
                    if stage > 0 and n_fade > 0 and epoch < n_fade:
                        progress = (epoch + i / max(1, len(train_loader))) / n_fade
                        alpha = float(max(0.0, min(1.0, progress)))
                    else:
                        alpha = 1.0

                    optimizer.zero_grad()
                    x_recon, mu, logvar = model(x, stage=stage, alpha=alpha)
                    loss, recon_loss, kld_loss = vae_bce_loss(x_tgt, x_recon, mu, logvar)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    recon_sum  += recon_loss.item()
                    kld_sum    += kld_loss.item()
                    global_step += 1

                n_samples = len(train_loader.dataset)
                print(f"[Stage {stage}] Epoch {epoch+1}/{n_epochs} {alpha_info} - "
                      f"Loss: {total_loss/n_samples:.4f}, Recon: {recon_sum/n_samples:.4f}, KLD: {kld_sum/n_samples:.4f}")

            # 保存每个阶段的sample
            model.eval()
            with torch.no_grad():
                z = torch.randn(64, args.z_dim).to(device)
                sample = model.decode(z, stage=stage, alpha=1.0)
                save_image(sample, os.path.join(exp_dir, f'sample_stage{stage}_{cur_res}x{cur_res}.png'), nrow=8)

        print("Finish training!")

        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(exp_dir, 'pro_vae.pth'))

        # 重建图像
        model.eval()
        with torch.no_grad():
            test_x, _ = next(iter(test_loader))
            test_x = test_x[:32].to(device)
            # 最终分辨率重建
            final_res = args.final_res
            x_tgt = F.interpolate(test_x, size=(final_res, final_res), mode="area")
            recon_x, _, _ = model(x_tgt, stage=model.max_stage, alpha=1.0)
            comparison = torch.cat([x_tgt[:16], recon_x[:16]])
            save_image(comparison, os.path.join(exp_dir, f'reconstruction_final_{final_res}.png'), nrow=8)
            print(f"结果已保存到 {exp_dir}")

            # FID
            fid = compute_fid_for_vae(model, test_loader, device,
                                      input_dim=final_res*final_res, z_dim=args.z_dim)
            print(f"FID: {fid:.4f}")


if __name__ == '__main__':
    main(parse_args())
