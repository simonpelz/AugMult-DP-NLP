import pytest
import torch
from torch import nn

from dpaug import mean_view_ce, per_example_grads
from dpaug.ghost_norm import linear_per_example_sq_norm


@pytest.mark.parametrize("method", ["ghost", "materialize", "auto"])
def test_ghost_identity_random(method):
    g = torch.Generator().manual_seed(0)
    n, k, t, din, dout = 3, 4, 5, 6, 7
    x = torch.randn(n, k, t, din, generator=g, dtype=torch.float64)
    b = torch.randn(n, k, t, dout, generator=g, dtype=torch.float64)
    dw = torch.einsum("nkto,nkti->noi", b, x)  # sum over views and tokens
    db = b.sum((1, 2))
    ref = dw.pow(2).sum((1, 2)) + db.pow(2).sum(1)
    torch.testing.assert_close(linear_per_example_sq_norm(x, b, bias=True, method=method), ref)
    torch.testing.assert_close(linear_per_example_sq_norm(x, b, bias=False, method=method), dw.pow(2).sum((1, 2)))


class Toy(nn.Module):
    def __init__(self, vocab=40, d=8, c=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.target = nn.Linear(d, 12)
        self.out = nn.Linear(12, c)

    def forward(self, ids):  # ids: (B, T)
        return self.out(torch.tanh(self.target(self.emb(ids))).mean(-2))


def test_ghost_norm_matches_augmult_per_example_grads_in_model():
    """Captured (activation, per-example backprop) of a Linear over K views -> ghost norm == vmap grad norm."""
    torch.manual_seed(0)
    n, k, t = 5, 3, 6
    model = Toy().double()
    ids = torch.randint(0, 40, (n, k, t))
    y = torch.randint(0, 3, (n,))

    # reference: materialised AugMult per-example grads of the target layer
    g = per_example_grads(model, ids, y)
    ref = g["target.weight"].pow(2).sum((1, 2)) + g["target.bias"].pow(2).sum(1)

    # ghost: one batched forward over (N*K, T); total loss = SUM over examples of the
    # per-example (mean-over-views) loss, so the captured output grads are per-example backprops.
    saved = {}
    h1 = model.target.register_forward_hook(lambda m, i, o: saved.__setitem__("x", i[0].detach()))
    h2 = model.target.register_full_backward_hook(lambda m, gi, go: saved.__setitem__("g", go[0].detach()))
    logits = model(ids.reshape(n * k, t)).reshape(n, k, -1)
    total = sum(mean_view_ce(logits[i], y[i]) for i in range(n))
    total.backward()
    h1.remove()
    h2.remove()
    x = saved["x"].reshape(n, k, t, -1)
    gb = saved["g"].reshape(n, k, t, -1)
    torch.testing.assert_close(linear_per_example_sq_norm(x, gb, method="ghost"), ref)
