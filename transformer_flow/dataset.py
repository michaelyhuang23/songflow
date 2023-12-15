import torch
from torch.nn import functional as F
import os 

class SpecDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_heads=8, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.pt')]
        self.o_D = torch.load(self.files[0]).T.shape[1]
        self.num_heads = num_heads
        self.T = self[0].shape[0]
        self.D = self[0].shape[1]
    
    def pad_tensor(self, spec):
        if spec.shape[1] % self.num_heads != 0:
            spec = F.pad(spec, (0, self.num_heads - (spec.shape[1] % self.num_heads)), 'constant', 0)
        return spec
    
    def revert_tensor(self, spec):
        return self.revert_normalize(spec[:, :self.o_D]).T

    def normalize(self, spec):
        return spec/100

    def revert_normalize(self, spec):
        return spec*100

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        spec = torch.load(self.files[idx]).T
        spec = self.normalize(self.pad_tensor(spec))
        if self.transform:
            spec = self.transform(spec)
        return spec
