import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.logvar_layer = nn.Linear(hidden_dim, z_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

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

def vae_loss_gaussian_mc(x, x_recon, mu, logvar, z):
    recon_loss = 0.5 * torch.sum(math.log(2 * math.pi) + logvar + torch.pow(x - x_recon, 2) / torch.exp(logvar))
    log_p_z = log_density_gaussian(z, torch.zeros_like(mu), torch.zeros_like(logvar))
    log_q_z_x = log_density_gaussian(z, mu, logvar)
    KLD = torch.sum(log_q_z_x - log_p_z)
    return recon_loss + KLD, recon_loss, KLD

# Frey Face: Gaussian likelihood + KLD (Analytical)
def vae_loss_gaussian_analytical(x, x_recon, mu, logvar):
    recon_loss = 0.5 * torch.sum(math.log(2 * math.pi) + logvar + torch.pow(x - x_recon, 2) / torch.exp(logvar))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD, recon_loss, KLD
