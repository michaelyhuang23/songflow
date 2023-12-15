import torch
from torch import nn
from torch.nn import functional as F
from model import MaskedTransformer, build_model
from dataset import SpecDataset
import os

model = MaskedTransformer(20, 1008, num_blocks=2, output_size=2*1000, nhead=8, dim_feedforward=128, activation=F.relu)
flow = build_model(T=20, D=1008, num_layers=2, dim_feedforward=128)

dataset = SpecDataset(os.path.join('../', 'square_jamendo_data'))
x = dataset[0][None, :20, :]
print(x.shape)
model(x)
with torch.no_grad():
    print(model(x))
    print(flow.transform_to_noise(x))