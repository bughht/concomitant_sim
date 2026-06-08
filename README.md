# concomitant_sim

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](LICENSE)

**`concomitant_sim`** is a fast, backend-agnostic Python package for simulating the phase accumulation caused by concomitant (Maxwell) fields in Magnetic Resonance Imaging (MRI).

It enables high-performance computations on both CPU and GPU by natively supporting NumPy, PyTorch, and CuPy arrays. The library uses an intelligent dispatching mechanism to adapt array operations to your inputs seamlessly, making it an ideal choice for regular simulation pipelines as well as differentiable deep learning MRI models.

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

## License

This project is licensed under the [WTFPL](LICENSE) (Do What The F*ck You Want To Public License).