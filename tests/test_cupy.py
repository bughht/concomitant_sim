import pytest

cupy = pytest.importorskip("cupy")
from concomitant_sim import concomitant_sim


def test_concomitant_cupy_shape():
    n_t = 16
    t = cupy.linspace(0.0, 1e-3, n_t)
    G = cupy.stack(
        [
            cupy.zeros(n_t),
            cupy.zeros(n_t),
            cupy.linspace(0.0, 1e3, n_t),
        ],
        axis=1,
    )

    n_grid = 25
    X, Y, Z = cupy.meshgrid(
        cupy.linspace(-0.15, 0.15, n_grid),
        cupy.linspace(-0.15, 0.15, n_grid),
        cupy.linspace(-0.15, 0.15, n_grid),
        indexing="xy",
    )
    r = cupy.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    phi = concomitant_sim(t, G, r)

    assert phi.shape == (n_grid ** 3, n_t)
