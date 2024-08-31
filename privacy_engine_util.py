from functools import partial
from typing import Sequence, Type, Union, List
import torch
import torch.nn as nn

from augmult_grad_sample_module import GradSampleModuleAugMult

from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.grad_sample.gsm_exp_weights import GradSampleModuleExpandedWeights
from opacus.grad_sample.gsm_no_op import GradSampleModuleNoOp

"""
!!!
To Integrate into opacus, just add these 2 lines to get_gsm_class() in grad_sample.utils, this is just a workaround

elif grad_sample_mode == "augmult":
    return GradSampleModuleAugMult
"""


def modified_get_gsm_class(grad_sample_mode: str) -> Type[AbstractGradSampleModule]:
    """
    Returns AbstractGradSampleModule subclass correspinding to the input mode.
    See README for detailed comparison between grad sample modes.

    :param grad_sample_mode:
    :return:
    """
    if grad_sample_mode in ["hooks", "functorch"]:
        return GradSampleModule
    elif grad_sample_mode == "ew":
        return GradSampleModuleExpandedWeights
    elif grad_sample_mode == "ghost":
        return GradSampleModuleFastGradientClipping
    elif grad_sample_mode == "no_op":
        return GradSampleModuleNoOp
    elif grad_sample_mode == "augmult":
        return GradSampleModuleAugMult
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Allowed values: hooks, functorch, ew, ghost, no_op, augmult"
        )

def wrap_model(model: nn.Module, grad_sample_mode: str, *args, **kwargs):
    cls = modified_get_gsm_class(grad_sample_mode)
    if grad_sample_mode == "functorch":
        kwargs["force_functorch"] = True
    return cls(model, *args, **kwargs)

def _prepare_model_modified(
        self,
        module: nn.Module,
        *,
        batch_first: bool = True,
        max_grad_norm: Union[float, List[float]] = 1.0,
        loss_reduction: str = "mean",
        grad_sample_mode: str = "hooks",
    ) -> AbstractGradSampleModule:
        
    """
    Identical copy from opacus PrivacyEngine, only difference is calling a modified get_gsm_class() to accomodate AugMultGradSampler
    """
    # Ideally, validation should have been taken care of by calling
    # `get_compatible_module()`
    self.validate(module=module, optimizer=None, data_loader=None)

    # wrap
    if isinstance(module, AbstractGradSampleModule):
        if (
            module.batch_first != batch_first
            or module.loss_reduction != loss_reduction
            or type(module) is not modified_get_gsm_class(grad_sample_mode)
        ):
            raise ValueError(
                f"Pre-existing GradSampleModule doesn't match new arguments."
                f"Got: module.batch_first: {module.batch_first}, module.loss_reduction: {module.loss_reduction}, type(module): {type(module)}"
                f"Requested: batch_first:{batch_first}, loss_reduction: {loss_reduction}, grad_sample_mode: {grad_sample_mode} "
                f"Please pass vanilla nn.Module instead"
            )

        return module
    else:
        if grad_sample_mode == "ghost":
            return wrap_model(
                module,
                grad_sample_mode=grad_sample_mode,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
                max_grad_norm=max_grad_norm,
            )
        else:
            return wrap_model(
                module,
                grad_sample_mode=grad_sample_mode,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
            )
        

def collate_dict(batch,*,collate_fn,sample_empty_shapes,dtypes,):
   
    if len(batch) > 0:
        return collate_fn(batch)
    else:
        return {key:
            torch.zeros(sample_empty_shapes[key], dtype=dtypes[key])
            for key in sample_empty_shapes
        }


def dict_wrap_collate_with_empty(*,collate_fn,sample_empty_shapes,dtypes,):
    return partial(
        collate_dict,
        collate_fn=collate_fn,
        sample_empty_shapes=sample_empty_shapes,
        dtypes=dtypes,
    )