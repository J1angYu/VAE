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
        return x_recon, mu, logvar


def vae_loss(x, x_recon, mu, logvar):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD
