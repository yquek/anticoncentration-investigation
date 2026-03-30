"""Benchmark different time-evolution approaches for 5x5."""

import numpy as np
import time as timer
from anticoncentration_investigation import (
    square_lattice,
    precompute_pauli_specs,
    build_hamiltonian_linop,
    krylov_expm_multiply,
)
from scipy.sparse.linalg import expm_multiply


def pr(*a, **kw):
    print(*a, **kw, flush=True)


nx, ny = 5, 5
k = 2
n_sites, adj = square_lattice(nx, ny)
d = 1 << n_sites

fm, sm, bp, num_paulis = precompute_pauli_specs(n_sites, adj, k)
rng = np.random.default_rng(42)
g = rng.standard_normal(num_paulis)
H_op = build_hamiltonian_linop(fm, sm, bp, g, n_sites)

psi0 = np.zeros(d, dtype=complex)
psi0[0] = 1.0

# Warm up numba
pr("Warming up numba...")
_n, _adj = square_lattice(2, 2)
_fm, _sm, _bp, _np = precompute_pauli_specs(_n, _adj, 2)
_g = rng.standard_normal(_np)
_H = build_hamiltonian_linop(_fm, _sm, _bp, _g, _n)
_H.matvec(np.ones(1 << _n, dtype=complex))
pr("Done.\n")

dt = 1.0

# --- Approach 1: Custom Lanczos with different m ---
for m in [15, 20, 25, 30, 40]:
    t0 = timer.time()
    psi = krylov_expm_multiply(H_op.matvec, psi0, dt, m=m)
    elapsed = timer.time() - t0
    norm = np.linalg.norm(psi)
    probs = np.abs(psi) ** 2
    cp = d * np.sum(probs ** 2)
    pr(f"Lanczos m={m:2d}: {elapsed:6.1f}s  |psi|={norm:.10f}  d*CP={cp:.4f}")

# --- Approach 2: scipy expm_multiply with LinearOperator ---
pr("")
A_op = -1j * H_op  # scipy needs -iH operator
t0 = timer.time()
psi_scipy = expm_multiply(dt * A_op, psi0)
elapsed = timer.time() - t0
norm = np.linalg.norm(psi_scipy)
probs = np.abs(psi_scipy) ** 2
cp = d * np.sum(probs ** 2)
pr(f"scipy expm_multiply: {elapsed:6.1f}s  |psi|={norm:.10f}  d*CP={cp:.4f}")

# --- Check agreement ---
pr("\nAgreement check (vs scipy):")
for m in [15, 20, 25, 30, 40]:
    psi_l = krylov_expm_multiply(H_op.matvec, psi0, dt, m=m)
    diff = np.linalg.norm(psi_l - psi_scipy) / np.linalg.norm(psi_scipy)
    pr(f"  Lanczos m={m:2d} vs scipy:  ||diff||/||psi|| = {diff:.2e}")
