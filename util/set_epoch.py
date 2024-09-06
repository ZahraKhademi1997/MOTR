import torch


class CustomSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)
    
    def set_epoch(self, epoch):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)