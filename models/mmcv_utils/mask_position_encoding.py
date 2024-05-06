# Copyright (c) Hikvision Research Institute. All rights reserved.
import math
import torch


class RelSinePositionalEncoding(nn.Module):
    """Relative Position encoding with sine and cosine functions.

    This is designed for use in transformers, as described in 'Attention is All You Need'.
    The implementation uses sine and cosine functions of different frequencies.

    Args:
        num_feats (int): The feature dimension for each position along x-axis or y-axis.
                         Note the final returned dimension for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position embedding. The scale will be used only when `normalize` is True.
        eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-6.
        offset (float, optional): Offset added to embed when do the normalization. Defaults to 0.
    """
    def __init__(self, num_feats, temperature=10000, normalize=False, scale=2*math.pi, eps=1e-6, offset=0.0):
        super(RelSinePositionalEncoding, self).__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask, coord):
        """Forward function for `RelSinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing ignored positions, while zero values means valid positions for this image. Shape [bs, h, w].
            coord (Tensor): Center coordinate of the current instance.

        Returns:
            pos (Tensor): Returned position embedding with shape [bs, num_feats*2, h, w].
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = ((y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) - coord[1]) * self.scale
            x_embed = ((x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) - coord[0]) * self.scale
        else:
            y_embed = y_embed - coord[1]
            x_embed = x_embed - coord[0]

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(num_feats={self.num_feats}, temperature={self.temperature}, "
        repr_str += f"normalize={self.normalize}, scale={self.scale}, eps={self.eps})"
        return repr_str
    
    
def build_positional_encoding(cfg):
    """Factory for building position encoding modules from a configuration dictionary."""
    positional_encoding_type = cfg['type']
    
    if positional_encoding_type == 'RelSinePositionalEncoding':
        return RelSinePositionalEncoding(
            num_feats=cfg['num_feats'],
            temperature=cfg.get('temperature', 10000),
            normalize=cfg['normalize'],
            scale=cfg.get('scale', 2 * math.pi),
            eps=cfg.get('eps', 1e-6),
            offset=cfg.get('offset', 0.0)
        )
    else:
        raise NotImplementedError(f"Positional encoding type '{positional_encoding_type}' is not implemented")
