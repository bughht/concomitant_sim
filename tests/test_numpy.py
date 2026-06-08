import pytest

np = pytest.importorskip("numpy")
from concomitant_sim import concomitant_sim


def test_concomitant_np_shape():
	n_t = 16
	t = np.linspace(0.0, 1e-3, n_t)
	G = np.stack(
		[
			np.zeros(n_t),
			np.zeros(n_t),
			np.linspace(0.0, 1e3, n_t),
		],
		axis=1,
	)

	n_grid = 25
	X, Y, Z = np.meshgrid(
		np.linspace(-0.15, 0.15, n_grid),
		np.linspace(-0.15, 0.15, n_grid),
		np.linspace(-0.15, 0.15, n_grid),
		indexing="ij",
	)
	r = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
	phi = concomitant_sim(t, G, r)

	assert phi.shape == (n_grid ** 3, n_t)
