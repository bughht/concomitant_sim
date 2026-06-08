import pytest

torch = pytest.importorskip("torch")
from concomitant_sim import concomitant_sim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_concomitant_torch_nograd_cuda():
	device = torch.device("cuda")

	n_t = 16
	t = torch.linspace(0.0, 1e-3, n_t, device=device)
	G = torch.stack(
		[
			torch.zeros(n_t, device=device),
			torch.zeros(n_t, device=device),
			torch.linspace(0.0, 1e3, n_t, device=device),
		],
		dim=1,
	)

	n_grid = 25
	X, Y, Z = torch.meshgrid(
		torch.linspace(-0.15, 0.15, n_grid, device=device),
		torch.linspace(-0.15, 0.15, n_grid, device=device),
		torch.linspace(-0.15, 0.15, n_grid, device=device),
		indexing="ij",
	)
	r = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

	with torch.no_grad():
		phi = concomitant_sim(t, G, r)

	assert phi.shape == (n_grid**3, n_t)
	assert phi.is_cuda
