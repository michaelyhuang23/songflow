from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows import transforms, distributions, flows
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nflows.utils import torchutils

class MaskedTransformer(nn.Module):
    def __init__(self, features_num, features_dim, num_blocks=2, output_size=2000, nhead=8, dim_feedforward=128, activation=F.relu):
        super(MaskedTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead, dim_feedforward=dim_feedforward, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_blocks)
        self.linear1 = nn.Linear(features_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, output_size)
        self.activation = activation
        self.mask = nn.Parameter(torch.triu(torch.ones(features_num, features_num), diagonal=1).bool(), requires_grad=False)

    def forward(self, x, context=None):
        x = self.activation(self.transformer_encoder(x, mask=self.mask.data))
        return self.linear2(self.linear1(x)) # this is a rather sketchy way of doing things


class MaskedAutoregresssiveAttentionTransform(AutoregressiveTransform):
    def __init__(
        self,
        features_num,
        features_dim,
        num_blocks=2,
        nhead=8,
        dim_feedforward=128,
        activation=F.relu,
    ):
        self.features_num = features_num
        self.features_dim = features_dim

        model = MaskedTransformer(features_num, features_dim, num_blocks=num_blocks, output_size=features_dim*self._output_dim_multiplier(), nhead=nhead, dim_feedforward=dim_feedforward, activation=activation)
        self._epsilon = 1e-3
        super(MaskedAutoregresssiveAttentionTransform, self).__init__(model)

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(
            -1, self.features_num, self.features_dim, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift


def build_model(T=200, D=1000, num_layers=10, dim_feedforward=128):
    # Define a sequence of transformations.
    # inputs will be shape (B, T, D) where we do autoregressive flow over T
    transform_list = []
    for i in range(num_layers):
        transform_list.append(MaskedAutoregresssiveAttentionTransform(features_num=T, features_dim=D, num_blocks=2, nhead=8, dim_feedforward=dim_feedforward, activation=F.relu))
        transform_list.append(transforms.ReversePermutation(features=T, dim=1)) # reverse along T
    transform = transforms.CompositeTransform(transform_list)

    # Define a base distribution.
    base_distribution = distributions.StandardNormal(shape=[T, D])

    # Combine into a flow.
    flow = flows.Flow(transform=transform, distribution=base_distribution)
    return flow