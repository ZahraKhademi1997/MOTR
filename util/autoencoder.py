import torch
import torch.nn as nn
import torch.optim as optim
# class BoundingBoxAutoencoder(nn.Module):
#     def __init__(self):
#         super(BoundingBoxAutoencoder, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.GELU(),
#             nn.Linear(128, 64),
#             nn.GELU(),
#             nn.Linear(64, 4)  # Compressed to 4D latent space
#         )
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(4, 64),
#             nn.GELU(),
#             nn.Linear(64, 128),
#             nn.GELU(),
#             nn.Linear(128, 4)  # Outputting reconstructed 4D bounding boxes
#         )
    
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded



class BoundingBoxAutoencoder(nn.Module):
    def __init__(self):
        super(BoundingBoxAutoencoder, self).__init__()
        
        # Encoder: Compress from (300, 256) to (300, 4)
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),  # (300, 256) -> (300, 128)
            nn.ReLU(True),
            nn.Linear(128, 64),   # (300, 128) -> (300, 64)
            nn.ReLU(True),
            nn.Linear(64, 16),    # (300, 64) -> (300, 16)
            nn.ReLU(True),
            nn.Linear(16, 4)      # (300, 16) -> (300, 4)
        )
        
        # Decoder: Reconstruct from (300, 4) to (300, 256)
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),     # (300, 4) -> (300, 16)
            nn.ReLU(True),
            nn.Linear(16, 64),    # (300, 16) -> (300, 64)
            nn.ReLU(True),
            nn.Linear(64, 128),   # (300, 64) -> (300, 128)
            nn.ReLU(True),
            nn.Linear(128, 256),  # (300, 128) -> (300, 256)
            nn.Sigmoid()          # Final layer to match the input range (0 to 1)
        )
    
    def forward(self, x):
        # Encode the input to latent space
        encoded = self.encoder(x)
        
        # Decode the latent space to reconstruct the original input
        decoded = self.decoder(encoded)
        
        return encoded, decoded
