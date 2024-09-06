import torch
from torch import nn


class BoundingBoxAutoencoder(nn.Module):
    def __init__(self):
        super(BoundingBoxAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4)  # Compressed to 4D latent space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 4)  # Outputting reconstructed 4D bounding boxes
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
