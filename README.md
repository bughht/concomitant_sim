# MRI Concomitant Field Simulation Tool

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](LICENSE)

**`concomitant_sim`** is a fast, backend-agnostic Python package for simulating the phase accumulation caused by concomitant fields (Maxwell Terms) in MRI.

It enables high-performance computations on both CPU and GPU by natively supporting NumPy, PyTorch, and CuPy arrays. The library uses an intelligent dispatching mechanism to adapt array operations to your inputs seamlessly, making it an ideal choice for regular simulation pipelines as well as differentiable deep learning MRI models.

## Theoretical Modeling
The phase accumulation $\Phi_c$ caused by concomitant fields is modeled by integrating the magnetic field variation $\Delta B_c$ over time $t$. Following the standard framework established by Bernstein et al. (1998), for a system with primary linear gradients, the concomitant field $B_c$ is approximated by second-order spatial terms:

The phase $\Phi_c$ at a position $\mathbf{r} = (x, y, z)$ is given by the time integral:
$$\Phi_c(\mathbf{r}, TE) = \gamma \int_0^{TE} \Delta B_c(\mathbf{r}, t) dt$$
Where $\Delta B_c$ is determined by the gradient waveforms:
$$\Delta B_c(x, y, z, t)  \approx \frac{1}{2B_0} \left[\left(G_x^2+G_y^2\right)z^2 + G_z^2\frac{x^2+y^2}{4} - G_xG_zxz - G_yG_zyz \right] $$

---

## ⚡ Key Features

- **Physics-Informed Simulation**: Calculates exact integration of 3D gradient waveform cross-terms (e.g., $xx, yy, zz, xz, yz$) combined with spatial polynomial coefficients to compute magnetic field variation ($\Delta B_c$).
- **Backend-Agnostic Engine**: Pass arrays from your array library of choice. The internal computations will transparently map to the correct backend.
- **Hardware Acceleration (GPU)**: Natively supports CuPy and PyTorch for incredibly fast GPU-accelerated massive spin simulations.
- **Auto-Grad Compatible**: If you provide PyTorch tensors requiring gradients (`requires_grad=True`), the library maintains the computational graph securely, unlocking backpropagation through your physics simulation.

---

## 📦 Installation

To install standard features (NumPy backend):

```bash
pip install .
```

To install with specific backend accelerations, use the optional dependencies provided in the `pyproject.toml`:

```bash
# PyTorch backend for autograd & GPU support
pip install .[torch]

# CuPy backend for raw GPU speed
pip install .[cupy]

# Development setup with testing suites
pip install .[dev]
```

---

## 🚀 Quickstart

For a complete walkthrough of the package, including how to use it with different backends (NumPy, CuPy, PyTorch) and track gradients, please refer to our example notebook:

👉 [examples/demo.ipynb](examples/demo.ipynb)

---

## Authors
- **Haotian Hong** - [hhong6@mgh.harvard.edu](mailto:hhong6@mgh.harvard.edu)

## References

**Bernstein, M. A., Zhou, X. J., Polzin, J. A., King, K. F., Ganin, A., Pelc, N. J., & Glover, G. H. (1998).** Concomitant gradient terms in phase contrast MR: Analysis and correction. *"Magnetic resonance in medicine" 39.2 (1998): 300-308.*

## License

This project is licensed under the [WTFPL](LICENSE) (Do What The F*ck You Want To Public License).