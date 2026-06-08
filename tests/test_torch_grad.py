import pytest

torch = pytest.importorskip("torch")
from concomitant_sim import concomitant_sim


def test_concomitant_torch_grad():
    n_t = 16
    t = torch.linspace(0.0, 1e-3, n_t)
    G = torch.stack(
        [
            torch.zeros(n_t),
            torch.zeros(n_t),
            torch.linspace(0.0, 1e3, n_t),
        ],
        dim=1,
    ).requires_grad_()

    n_grid = 25
    X, Y, Z = torch.meshgrid(
        torch.linspace(-0.15, 0.15, n_grid),
        torch.linspace(-0.15, 0.15, n_grid),
        torch.linspace(-0.15, 0.15, n_grid),
        indexing="ij",
    )
    r = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).requires_grad_()
    phi = concomitant_sim(t, G, r)

    loss = phi.sum()
    loss.backward()

    assert G.grad is not None
    assert phi.shape == (n_grid**3, n_t)