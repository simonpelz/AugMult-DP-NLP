#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
from opacus.utils.tensor_utils import sum_over_all_but_batch_and_last_n
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
        backprops = backprops.reshape((-1,self.K,)+ (backprops.shape[1:]))

        if layer.weight.requires_grad:
            normalize_activations = F.layer_norm(activations, layer.normalized_shape, eps=layer.eps)
            normalize_activations = normalize_activations.reshape((-1,self.K,)+ (activations.shape[1:]))
            ret[layer.weight] = sum_over_all_but_batch_and_last_n(
                normalize_activations
                * backprops,
                layer.weight.dim(),
            )
        if layer.bias.requires_grad:
            ret[layer.bias] = sum_over_all_but_batch_and_last_n(backprops, layer.bias.dim())
        return ret