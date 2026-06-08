from __future__ import annotations

from .backend import infer_backend, asarray, zeros, cumsum, concatenate


def concomitant_sim(t, G, r, B0: float = 2.89):
    """
    Backend-agnostic concomitant field phase simulation.

    Parameters
    ----------
    t : array-like, shape [Nt]
    G : array-like, shape [Nt, 3]
    r : array-like, shape [Nr, 3]
    B0 : float

    Returns
    -------
    phase : array-like, shape [Nr, Nt]
    """
    backend = infer_backend(t, G, r)
    xp = backend.lib

    t = asarray(t, backend)
    G = asarray(G, backend)
    r = asarray(r, backend)

    gamma_rad = 2.0 * xp.pi * 42.576e6

    x = r[:, 0:1]
    y = r[:, 1:2]
    z = r[:, 2:3]

    coeff_xx_yy = z**2
    coeff_zz = 0.25 * (x**2 + y**2)
    coeff_xz = x * z
    coeff_yz = y * z

    dt = t[1:] - t[:-1]

    def integrate_pair_product(A, B, dt):
        An, An1 = A[:-1], A[1:]
        Bn, Bn1 = B[:-1], B[1:]
        return (dt / 3.0) * (
            An * Bn + An1 * Bn1 + 0.5 * (An * Bn1 + An1 * Bn)
        )

    int_xx = integrate_pair_product(G[:, 0], G[:, 0], dt)
    int_yy = integrate_pair_product(G[:, 1], G[:, 1], dt)
    int_zz = integrate_pair_product(G[:, 2], G[:, 2], dt)
    int_xz = integrate_pair_product(G[:, 0], G[:, 2], dt)
    int_yz = integrate_pair_product(G[:, 1], G[:, 2], dt)

    delta_Bc = (1.0 / (2.0 * B0)) * (
        (int_xx + int_yy) * coeff_xx_yy
        + int_zz * coeff_zz
        - int_xz * coeff_xz
        - int_yz * coeff_yz
    )

    phase_inc = gamma_rad * delta_Bc
    phase = cumsum(phase_inc, axis=1, backend=backend)
    zero_col = zeros((r.shape[0], 1), backend=backend, dtype=phase.dtype)
    return concatenate([zero_col, phase], axis=1, backend=backend)