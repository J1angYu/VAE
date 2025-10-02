import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_VAE(nn.Module):
    def __init__(self, z_dim=20):
        super().__init__()
        # Encoder: 输入 [B,1,28,28] -> feature -> μ, logvar
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # [B,32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # [B,64,7,7]
            nn.ReLU(),
        )
        self.enc_out_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_out_dim, z_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, z_dim)

        # Decoder: z -> feature map -> x'
        self.fc_decode = nn.Linear(z_dim, self.enc_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # [B,32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # [B,1,28,28]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


import math

def log_density_gaussian(x, mu, logvar):
    """
    计算高斯分布在x点的对数概率密度。
    """
    norm_const = -0.5 * (math.log(2 * math.pi) + logvar)
    dist_term = -0.5 * ((x - mu) ** 2) / torch.exp(logvar)
    return norm_const + dist_term

def vae_loss_mc(x, x_recon, mu, logvar, z):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # 计算 log p(z), p(z) 是 N(0, I)
    log_p_z = log_density_gaussian(z, torch.zeros_like(mu), torch.zeros_like(logvar))
    
    # 计算 log q(z|x), q(z|x) 是 N(mu, logvar)
    log_q_z_x = log_density_gaussian(z, mu, logvar)
    
    # KLD ≈ log q(z|x) - log p(z)
    KLD = torch.sum(log_q_z_x - log_p_z)
    
    return BCE + KLD, BCE, KLD

def vae_loss_analytical(x, x_recon, mu, logvar):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD
