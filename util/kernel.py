import torch
import torch.nn as nn

class RBFKernelTransformer(nn.Module):
    def __init__(self, feature_dim, num_landmarks, gamma):
        super(RBFKernelTransformer, self).__init__()
        # Initialize landmarks randomly, or load from a predefined set
        self.landmarks = nn.Parameter(torch.rand(num_landmarks, feature_dim), requires_grad=False)
        self.gamma = gamma

    def rbf_kernel_feature(self, data):
        """
        Apply RBF kernel to map data to a higher-dimensional space using predefined landmarks.
        """
        data_sq = torch.sum(data ** 2, axis=1, keepdim=True)
        landmarks_sq = torch.sum(self.landmarks ** 2, axis=1)
        cross_term = 2 * torch.matmul(data, self.landmarks.t())
        sdist = data_sq + landmarks_sq - cross_term
        return torch.exp(-self.gamma * sdist)

    def forward(self, data):
        """
        Forward pass through the RBF kernel transformer.
        """
        return self.rbf_kernel_feature(data)