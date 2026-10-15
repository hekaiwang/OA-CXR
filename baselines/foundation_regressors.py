"""Fixed task-adapted regressors for independent frozen encoders and TabM.

The three visual encoders use the exact OA-CXR residual readout implementation.
TabM uses its official implementation, without a substitute architecture. All
models train on raw predictions and clamp only their inference score.
"""
from __future__ import annotations

import torch
from torch import nn

from oa_cxr.rebuild.neural_readout import FitStandardizer, NeuralReadout

METHODS = ('biomedclip', 'chexzero', 'dinov2', 'tabm')


class TabMRegressor(nn.Module):
    def __init__(self, input_dim, *, k=32, d_block=512, n_blocks=3):
        super().__init__()
        from tabm import TabM
        self.input_dim, self.k = input_dim, k
        self.d_block, self.n_blocks = d_block, n_blocks
        self.standardizer = FitStandardizer(input_dim)
        self.model = TabM.make(n_num_features=input_dim, cat_cardinalities=None,
            d_out=1, k=k, d_block=d_block, n_blocks=n_blocks,
            num_embeddings=None, dropout=0.1, activation='ReLU', arch_type='tabm')

    def fit_standardizer(self, batches, *, split):
        self.standardizer.fit(batches, split=split)

    def forward(self, features):
        raw = self.model(self.standardizer(features)).squeeze(-1)
        if raw.ndim != 2 or raw.shape != (len(features), self.k) or not torch.isfinite(raw).all():
            raise ValueError('Official TabM must return finite [batch, ensemble] regression predictions')
        mean = raw.mean(dim=1)
        return dict(raw_prediction=raw, loss_prediction=raw, retention=mean.clamp(0, 1))

    def config(self):
        return dict(input_dim=self.input_dim, k=self.k, d_block=self.d_block, n_blocks=self.n_blocks,
            dropout=0.1, activation='ReLU', arch_type='tabm', start_scaling_init='random-signs', numerical_embeddings=None)


def make_regressor(method, input_dim):
    if method not in METHODS:
        raise ValueError('Unregistered foundation baseline')
    if method == 'tabm':
        return TabMRegressor(input_dim)
    return NeuralReadout(input_dim, width=256, inner_width=256, blocks=2,
        dropout=0.0, activation='linear_clamp', residual_mode='none')


def raw_l1_loss(output, target):
    prediction = output['loss_prediction']
    if prediction.ndim == 2:
        if prediction.shape[0] != len(target):
            raise ValueError('Memberwise regression batch differs')
        return (prediction - target[:, None]).abs().mean()
    if prediction.shape != target.shape:
        raise ValueError('Regression target shape differs')
    return (prediction - target).abs().mean()
