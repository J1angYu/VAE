import torch
import torch.nn.functional as F

def vae_bce_loss(x, x_recon, mu, logvar):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


def gaussian_likelihood_loss(x, x_recon, sigma=1.0):
    const = 0.5 * torch.log(2 * torch.pi * torch.tensor(sigma**2))
    reconstruction_error = (x - x_recon) ** 2
    gaussian_loss = torch.sum(reconstruction_error / (2 * sigma**2) + const)
    return gaussian_loss

def vae_gaussian_loss(x, x_recon, mu, logvar, sigma=1.0):
    # 高斯似然重构损失
    recon_loss = gaussian_likelihood_loss(x, x_recon, sigma)
    # KL散度损失
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + KLD
    return total_loss, recon_loss, KLD