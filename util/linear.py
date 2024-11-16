import math

import torch
import torch.nn as nn


if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
    
    
def obsolete_torch_version(torch_version, version_threshold) -> bool:
    return torch_version == 'parrots' or torch_version <= version_threshold

class Linear(torch.nn.Linear):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # empty tensor forward of Linear layer is supported in Pytorch 1.6
        if obsolete_torch_version(TORCH_VERSION, (1, 5)) and x.numel() == 0:
            out_shape = [x.shape[0], self.out_features]
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)