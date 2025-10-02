import os
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.provae import ProVAE_FadeIn, Res_ProVAE, elbo_bce_logits

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_to_32():
    # MNIST 28x28 -> 32x32
    return transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])

def resolution_for_stage(stage):
    return 4 * (2 ** stage)

def save_examples(model, device, out_dir, stage, alpha, z_dim, n=64, mode='fadein'):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(n, z_dim, device=device)
        if mode == 'fadein':
            logits = model.decoder(z, stage, alpha)
        else:
            logits = model.decoder(z, stage)
        samples = torch.sigmoid(logits)
        save_image(samples, os.path.join(out_dir, f'stage{stage}_alpha{alpha:.2f}_samples.png'),
                   nrow=int(n**0.5))

def train_one_epoch(model, loader, optimizer, device, stage, alpha, beta_kl=1.0):
    model.train()
    total = recon = kl = 0.0
    n_pix = 0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(x, stage, alpha)
        loss, r, k = elbo_bce_logits(x, logits, mu, logvar)
        # KL 退火：用 beta_kl 调节
        loss = r + beta_kl * k
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += (r + k).item()
        recon += r.item()
        kl += k.item()
        n_pix += bs * x.size(-1) * x.size(-2)
    # 返回按“每像素”的均值，便于不同分辨率可比
    return total / n_pix, recon / n_pix, kl / n_pix

def make_loader(root, batch_size, device):
    tfm = pad_to_32()  # 始终 32×32；低分辨阶段在 batch 里再下采样
    ds_train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == 'cuda' else {}
    return DataLoader(ds_train, batch_size=batch_size, shuffle=True, **kwargs)

def adjust_input_for_stage(x, stage):
    # 输入 x 是 32×32；低分辨阶段下采样到当前分辨率
    target_res = resolution_for_stage(stage)
    if x.size(-1) != target_res:
        x = torch.nn.functional.interpolate(x, size=(target_res, target_res), mode='nearest')
    return x

def main():
    ap = argparse.ArgumentParser(description='ProVAE (fade-in) vs Res-ProVAE on MNIST')
    ap.add_argument('--mode', type=str, default='fadein', choices=['fadein','residual'], help='模型：fadein 或 residual')
    ap.add_argument('--data', type=str, default='./data', help='数据目录')
    ap.add_argument('--out', type=str, default='./runs/provae_mnist', help='输出目录')
    ap.add_argument('--z_dim', type=int, default=16)
    ap.add_argument('--base_ch', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max_stage', type=int, default=3, help='0..3 对应 4,8,16,32')
    ap.add_argument('--epochs_transition', type=int, default=3, help='每个阶段的过渡期 epoch 数（仅 fade-in 用）')
    ap.add_argument('--epochs_stable', type=int, default=2, help='每个阶段的稳定期 epoch 数（fade-in）；residual 则两者相加')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    if args.mode == 'fadein':
        model = ProVAE_FadeIn(in_ch=1, z_dim=args.z_dim, base_ch=args.base_ch, max_stage=args.max_stage).to(device)
    else:
        model = Res_ProVAE(in_ch=1, z_dim=args.z_dim, base_ch=args.base_ch, max_stage=args.max_stage).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    # 数据加载（总是 32×32；按 stage 再下采样）
    loader = make_loader(args.data, args.batch_size, device)

    print(f"Training {args.mode} model on device={device} ...")
    for stage in range(0, args.max_stage + 1):
        res = resolution_for_stage(stage)
        print(f"== Stage {stage} | resolution {res}x{res} ==")

        if args.mode == 'fadein':
            # 过渡期：alpha 0→1，并把 KL 权重 beta_kl 同步从 0→1（可改）
            for ep in range(args.epochs_transition):
                alpha = (ep + 1) / max(1, args.epochs_transition)

                def gen_batches():
                    for x, y in loader:
                        x = adjust_input_for_stage(x, stage)
                        yield x, y

                total, rec, k = train_one_epoch(model, gen_batches(), opt, device, stage, alpha, beta_kl=alpha)
                print(f"[Transition] epoch {ep+1}/{args.epochs_transition} | alpha={alpha:.2f} | loss(px) {total:.4f} | recon {rec:.4f} | KL {k:.4f}")

            save_examples(model, device, args.out, stage, alpha=1.0, z_dim=args.z_dim, mode='fadein')

            # 稳定期：alpha=1
            for ep in range(args.epochs_stable):
                def gen_batches():
                    for x, y in loader:
                        x = adjust_input_for_stage(x, stage)
                        yield x, y

                total, rec, k = train_one_epoch(model, gen_batches(), opt, device, stage, alpha=1.0, beta_kl=1.0)
                print(f"[Stabilize] epoch {ep+1}/{args.epochs_stable} | alpha=1.00 | loss(px) {total:.4f} | recon {rec:.4f} | KL {k:.4f}")

            save_examples(model, device, args.out, stage, alpha=1.0, z_dim=args.z_dim, mode='fadein')

        else:
            # 残差模型：不需要 alpha，直接训（过渡+稳定的总 epoch 数）
            total_epochs = args.epochs_transition + args.epochs_stable
            for ep in range(total_epochs):
                def gen_batches():
                    for x, y in loader:
                        x = adjust_input_for_stage(x, stage)
                        yield x, y

                total, rec, k = train_one_epoch(model, gen_batches(), opt, device, stage, alpha=1.0, beta_kl=1.0)
                print(f"[Residual] epoch {ep+1}/{total_epochs} | loss(px) {total:.4f} | recon {rec:.4f} | KL {k:.4f}")

            save_examples(model, device, args.out, stage, alpha=1.0, z_dim=args.z_dim, mode='residual')

    # 保存模型
    torch.save({'model': model.state_dict(), 'args': vars(args)},
               os.path.join(args.out, f'{args.mode}_final.pt'))
    print("Done. Artifacts saved to:", args.out)

if __name__ == '__main__':
    main()
