#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.tensor_utils import unfold2d, unfold3d, sum_over_all_but_batch_and_last_n
import torch.nn.functional as F


class AugmentationMultiplicity:


    def __init__(self, K):
        self.K = K


    def compute_linear_grad_sample(
        self, layer: nn.Linear, activations: torch.Tensor, backprops: torch.Tensor
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for ``nn.Linear`` layer
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        activations = activations.reshape(
            (
                -1,
                self.K,
            )
            + (activations.shape[1:])
        )
        backprops = backprops.reshape(
            (
                -1,
                self.K,
            )
            + (backprops.shape[1:])
        )
        if layer.weight.requires_grad:
            gs = torch.einsum("n...i,n...j->nij", backprops, activations)
            ret[layer.weight] = gs
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("n...k->nk", backprops)
        return ret


    def compute_layer_norm_grad_sample(self,
        layer: nn.LayerNorm,
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for LayerNorm
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        if layer.weight.requires_grad:
            normalize_activations = F.layer_norm(activations, layer.normalized_shape, eps=layer.eps)
            normalize_activations = normalize_activations.reshape((-1,self.K,)+ (activations.shape[1:]))
            backprops = backprops.reshape((-1,self.K,)+ (backprops.shape[1:]))
            ret[layer.weight] = sum_over_all_but_batch_and_last_n(
                normalize_activations
                * backprops,
                layer.weight.dim(),
            )
        if layer.bias.requires_grad:
            ret[layer.bias] = sum_over_all_but_batch_and_last_n(backprops, layer.bias.dim())
        return ret


    def compute_embedding_grad_sample(
        self,
        layer: nn.Embedding,
        activations: torch.Tensor,
        backprops: torch.Tensor
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for ``nn.Embedding`` layer.

        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        if layer.weight.requires_grad:
            saved = torch.backends.cudnn.deterministic
            torch.backends.cudnn.deterministic = True

            # Reshape activations and backprops to handle augmentation multiplicity
            activations = activations.reshape((-1, self.K) + activations.shape[1:])
            backprops = backprops.reshape((-1, self.K) + backprops.shape[1:])

            batch_size = activations.shape[0]
            if batch_size == 0:
                ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
                return ret

            # Adjust index and gradient sample computation to account for augmentations
            index = (
                activations.unsqueeze(-1)
                .expand(*activations.shape, layer.embedding_dim)
                .reshape(batch_size, -1, layer.embedding_dim)
            )
            grad_sample = torch.zeros(
                batch_size, *layer.weight.shape, device=layer.weight.device
            )
            grad_sample.scatter_add_(
                1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
            )

            torch.backends.cudnn.deterministic = saved
            ret[layer.weight] = grad_sample

        return ret

    def original_non_augmult_compute_embedding_grad_sample(self,
        layer: nn.Embedding, activations: torch.Tensor, backprops: torch.Tensor
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for ``nn.Embedding`` layer.

        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        activations = activations[0]
        ret = {}
        if layer.weight.requires_grad:
            saved = torch.backends.cudnn.deterministic
            torch.backends.cudnn.deterministic = True

            batch_size = activations.shape[0]
            if batch_size == 0:
                ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
                return ret

            index = (
                activations.unsqueeze(-1)
                .expand(*activations.shape, layer.embedding_dim)
                .reshape(batch_size, -1, layer.embedding_dim)
            )
            grad_sample = torch.zeros(
                batch_size, *layer.weight.shape, device=layer.weight.device
            )
            grad_sample.scatter_add_(
                1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
            )
            torch.backends.cudnn.deterministic = saved
            ret[layer.weight] = grad_sample
        return ret