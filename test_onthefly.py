"""
Compare the on-the-fly Krylov approach against the original dense/sparse
approach on small lattices (2x2, 3x3).

For each lattice and each random Hamiltonian sample, we compare the
normalised collision probability d*CP(t) at times t = 0, 1, 2, 3, 4, 5, 6.

Tolerances: relative error 2e-3, absolute error 1e-3.
"""

import numpy as np
import sys

from anticoncentration_investigation import (
    square_lattice,
    enumerate_connected_subsets,
    precompute_paulis,
    precompute_pauli_specs,
    build_sparse_H,
    build_hamiltonian_linop,
    krylov_expm_multiply,
)
from scipy.sparse.linalg import expm_multiply


def collision_probability(psi):
    """d * sum_x |<x|psi>|^4"""
    probs = np.abs(psi) ** 2
    return len(psi) * np.sum(probs ** 2)


def run_dense(n_sites, adj, k, g, times):
    """Original dense/sparse approach: returns d*CP at each time."""
    d = 1 << n_sites
    phases, rows_flat, cols_flat, num_paulis = precompute_paulis(n_sites, adj, k)
    H = build_sparse_H(phases, rows_flat, cols_flat, g, d)
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0

    cp_vals = np.empty(len(times))
    for i, t in enumerate(times):
        if t == 0.0:
            psi = psi0.copy()
        else:
            psi = expm_multiply(-1j * t * H, psi0)
        cp_vals[i] = collision_probability(psi)
    return cp_vals


def run_onthefly(n_sites, adj, k, g, times, lanczos_m=30):
    """On-the-fly Krylov approach: returns d*CP at each time."""
    d = 1 << n_sites
    flip_masks, sign_masks, base_phases, num_paulis = precompute_pauli_specs(
        n_sites, adj, k
    )
    H_op = build_hamiltonian_linop(flip_masks, sign_masks, base_phases, g, n_sites)
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0

    cp_vals = np.empty(len(times))
    psi = psi0.copy()
    for i, t in enumerate(times):
        t_prev = times[i - 1] if i > 0 else 0.0
        dt = t - t_prev
        if dt > 0:
            psi = krylov_expm_multiply(
                H_op.matvec, psi, dt, m=lanczos_m,
            )
        cp_vals[i] = collision_probability(psi)
    return cp_vals


def test_shape(nx, ny, k=2, num_samples=5, seed=123, lanczos_m=30):
    n_sites, adj = square_lattice(nx, ny)
    d = 1 << n_sites
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rng = np.random.default_rng(seed)

    subsets = enumerate_connected_subsets(n_sites, adj, k)
    num_paulis = sum(3 ** len(s) for s in subsets)

    print(f"\n{'='*60}")
    print(f"Testing {nx}x{ny}  (n={n_sites}, d={d}, |P_k|={num_paulis})")
    print(f"{'='*60}")

    max_rel_err = 0.0
    max_abs_err = 0.0
    all_passed = True

    for s in range(num_samples):
        g = rng.standard_normal(num_paulis)
        cp_dense = run_dense(n_sites, adj, k, g, times)
        cp_otf = run_onthefly(n_sites, adj, k, g, times, lanczos_m=lanczos_m)

        for i, t in enumerate(times):
            abs_err = abs(cp_dense[i] - cp_otf[i])
            rel_err = abs_err / max(abs(cp_dense[i]), 1e-30)
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err

            passed = rel_err < 2e-3 or abs_err < 1e-3
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            print(f"  sample={s} t={t:.0f}  dense={cp_dense[i]:.6f}  "
                  f"otf={cp_otf[i]:.6f}  abs={abs_err:.2e}  "
                  f"rel={rel_err:.2e}  [{status}]")

    print(f"\nMax abs error: {max_abs_err:.2e}")
    print(f"Max rel error: {max_rel_err:.2e}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    lanczos_m = 30
    if len(sys.argv) > 1:
        lanczos_m = int(sys.argv[1])

    ok_2x2 = test_shape(2, 2, lanczos_m=lanczos_m)
    ok_3x3 = test_shape(3, 3, lanczos_m=lanczos_m)

    print(f"\n{'='*60}")
    print(f"2x2: {'PASS' if ok_2x2 else 'FAIL'}")
    print(f"3x3: {'PASS' if ok_3x3 else 'FAIL'}")
    print(f"{'='*60}")

    if not (ok_2x2 and ok_3x3):
        sys.exit(1)
