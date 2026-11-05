"""Reference per-example gradient norms for Linear layers under AugMult ("ghost AugMult").

For a Linear layer applied to a token sequence, the per-example weight gradient is
``dW = sum_t g_t x_t^T`` (g = backprop of the per-example loss w.r.t. the layer
output, x = layer input). Its squared Frobenius norm needs no materialisation:

    ||dW||_F^2 = sum_{t,t'} (x_t . x_t') (g_t . g_t') = <X X^T, G G^T>_F       (ghost clipping)

Under AugMult the per-example gradient is the gradient of a loss over K views, i.e.
``dW = sum_k sum_t g_{k,t} x_{k,t}^T``. That is exactly the single-sequence formula
on the *concatenated* token axis of length K*T — so ghost clipping (Li et al. 2022;
Bu et al. book-keeping) extends to AugMult by reshaping ``(N, K, T, d) -> (N, K*T, d)``.

Scaling: the Gram route costs O((KT)^2 (d_in + d_out)) vs O(KT d_in d_out) to
materialise; ``method="auto"`` picks the cheaper one per layer, as in mixed ghost clipping.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _flatten_views(t: Tensor) -> Tensor:
    # (N, K, T, d) -> (N, K*T, d); also accepts (N, T, d)
    if t.dim() == 4:
        n, k, s, d = t.shape
        return t.reshape(n, k * s, d)
    if t.dim() == 3:
        return t
    raise ValueError(f"expected (N,K,T,d) or (N,T,d), got {tuple(t.shape)}")


def linear_per_example_sq_norm(
    activations: Tensor, backprops: Tensor, bias: bool = True, method: str = "auto"
) -> Tensor:
    """Squared per-example grad norm of a Linear layer's (weight [+ bias]) over K views.

    ``activations``: ``(N, K, T, d_in)``; ``backprops``: ``(N, K, T, d_out)`` — the
    gradient of the *per-example* loss (not the batch-mean loss) w.r.t. the layer output.
    """
    x = _flatten_views(activations)
    g = _flatten_views(backprops)
    s, d_in, d_out = x.shape[1], x.shape[2], g.shape[2]
    if method == "auto":
        method = "ghost" if s * s * (d_in + d_out) < s * d_in * d_out else "materialize"
    if method == "ghost":
        gram_x = torch.einsum("nid,njd->nij", x, x)
        gram_g = torch.einsum("nid,njd->nij", g, g)
        sq = (gram_x * gram_g).sum((1, 2))
    elif method == "materialize":
        sq = torch.einsum("nto,nti->noi", g, x).pow(2).sum((1, 2))
    else:
        raise ValueError(method)
    if bias:
        sq = sq + g.sum(1).pow(2).sum(1)
    return sq
