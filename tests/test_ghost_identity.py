"""Ghost-clipping identity for AugMult (pure tensors, no Opacus).

For a Linear layer, the AugMult per-example weight gradient is
    G_i = sum_k sum_t b_{i,k,t} a_{i,k,t}^T
Treating the K views as one long sequence of K*T tokens, its squared Frobenius norm
can be computed without materializing G_i (Gram / "ghost norm" trick):
    ||G_i||_F^2 = sum_{s,s'} <a_s, a_s'> <b_s, b_s'>,   s, s' over the K*T tokens.
So ghost clipping supports AugMult by reshaping (N, K, T, d) -> (N, K*T, d).
"""

import torch


def materialized_sq_norm(a, b):
    # a: (N, K, T, d_in), b: (N, K, T, d_out)
    g = torch.einsum("nkto,nkti->noi", b, a)
    return (g**2).sum(dim=(1, 2))


def ghost_sq_norm(a, b):
    n = a.shape[0]
    a2 = a.reshape(n, -1, a.shape[-1])  # (N, K*T, d_in)
    b2 = b.reshape(n, -1, b.shape[-1])  # (N, K*T, d_out)
    ga = torch.bmm(a2, a2.transpose(1, 2))
    gb = torch.bmm(b2, b2.transpose(1, 2))
    return (ga * gb).sum(dim=(1, 2))


def test_ghost_norm_equals_materialized_norm_over_concatenated_views():
    torch.manual_seed(0)
    for n, k, t, din, dout in [(3, 1, 5, 7, 4), (2, 4, 6, 8, 3), (4, 16, 3, 5, 5)]:
        a = torch.randn(n, k, t, din, dtype=torch.float64)
        b = torch.randn(n, k, t, dout, dtype=torch.float64)
        torch.testing.assert_close(ghost_sq_norm(a, b), materialized_sq_norm(a, b))


def test_bias_ghost_norm():
    torch.manual_seed(1)
    b = torch.randn(3, 4, 6, 5, dtype=torch.float64)
    direct = (b.sum(dim=(1, 2)) ** 2).sum(-1)
    b2 = b.reshape(3, -1, 5)
    ghost = torch.bmm(b2, b2.transpose(1, 2)).sum(dim=(1, 2))
    torch.testing.assert_close(ghost, direct)


def test_view_sum_norm_differs_from_sum_of_view_norms():
    """Sanity: AugMult clips the norm of the view-sum, not the per-view norms."""
    torch.manual_seed(2)
    a = torch.randn(1, 4, 3, 6, dtype=torch.float64)
    b = torch.randn(1, 4, 3, 2, dtype=torch.float64)
    per_view = sum(materialized_sq_norm(a[:, j : j + 1], b[:, j : j + 1]) for j in range(4))
    assert not torch.allclose(per_view, ghost_sq_norm(a, b))
