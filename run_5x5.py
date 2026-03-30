"""Run anticoncentration simulation for 5x5 lattice using on-the-fly Krylov."""

import numpy as np
import json
import time as timer
from anticoncentration_investigation import (
    square_lattice,
    precompute_pauli_specs,
    build_hamiltonian_linop,
    krylov_expm_multiply,
)


def pr(*a, **kw):
    print(*a, **kw, flush=True)


def main():
    nx, ny = 5, 5
    k = 2
    seed = 42
    lanczos_m = 40
    num_samples = 5
    times = np.arange(0, 21, dtype=float)  # t = 0, 1, ..., 20

    n_sites, adj = square_lattice(nx, ny)
    d = 1 << n_sites
    T = len(times)
    rng = np.random.default_rng(seed)

    pr(f"Lattice: {nx}x{ny}, n={n_sites}, d={d:,}")
    pr(f"Times: {times[0]:.0f} to {times[-1]:.0f} ({T} points)")
    pr(f"Lanczos m={lanczos_m}, samples={num_samples}")

    pr("\nPrecomputing Pauli specs...")
    t0 = timer.time()
    flip_masks, sign_masks, base_phases, num_paulis = precompute_pauli_specs(
        n_sites, adj, k
    )
    pr(f"  |P_k| = {num_paulis}, took {timer.time() - t0:.2f}s")

    # Benchmark a single matvec
    pr("\nBenchmarking single matvec...")
    g_test = rng.standard_normal(num_paulis)
    H_test = build_hamiltonian_linop(flip_masks, sign_masks, base_phases, g_test, n_sites)
    v_test = np.random.randn(d) + 1j * np.random.randn(d)
    v_test /= np.linalg.norm(v_test)

    t0 = timer.time()
    _ = H_test.matvec(v_test)
    matvec_s = timer.time() - t0
    pr(f"  Single matvec: {matvec_s:.2f}s")
    pr(f"  Estimated per Lanczos step (m={lanczos_m}): ~{matvec_s:.1f}s matvec + reorth")
    pr(f"  Estimated per time step: ~{matvec_s * lanczos_m:.0f}s")
    pr(f"  Estimated per sample ({T} steps): ~{matvec_s * lanczos_m * T:.0f}s")
    pr(f"  Estimated total ({num_samples} samples): ~{matvec_s * lanczos_m * T * num_samples:.0f}s")

    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    cp_all = np.zeros((num_samples, T))

    t_total = timer.time()
    for s in range(num_samples):
        g = rng.standard_normal(num_paulis)
        H_op = build_hamiltonian_linop(
            flip_masks, sign_masks, base_phases, g, n_sites
        )
        psi = psi0.copy()

        t_sample = timer.time()
        for t_idx in range(T):
            t_now = times[t_idx]
            t_prev = times[t_idx - 1] if t_idx > 0 else 0.0
            dt = t_now - t_prev
            if dt > 0:
                psi = krylov_expm_multiply(H_op.matvec, psi, dt, m=lanczos_m)
            probs = np.abs(psi) ** 2
            cp_all[s, t_idx] = np.sum(probs ** 2)

            norm = np.linalg.norm(psi)
            pr(f"  sample {s+1}/{num_samples}  t={t_now:.0f}  "
               f"d*CP={d * cp_all[s, t_idx]:.4f}  |psi|={norm:.10f}  "
               f"({timer.time() - t_sample:.1f}s)")

        pr(f"  Sample {s+1} done in {timer.time() - t_sample:.1f}s")

    total_s = timer.time() - t_total
    pr(f"\nAll samples done in {total_s:.1f}s")

    norm_cp = d * np.mean(cp_all, axis=0)
    norm_cp_err = d * np.std(cp_all, axis=0) / np.sqrt(num_samples)

    pr(f"\n{'t':>4s}  {'d*CP':>10s}  {'err':>10s}")
    pr("-" * 30)
    for i, t in enumerate(times):
        pr(f"{t:4.0f}  {norm_cp[i]:10.4f}  {norm_cp_err[i]:10.4f}")

    results = {
        "lattice": f"{nx}x{ny}",
        "n_sites": n_sites,
        "d": d,
        "k": k,
        "num_paulis": num_paulis,
        "seed": seed,
        "lanczos_m": lanczos_m,
        "num_samples": num_samples,
        "times": times.tolist(),
        "norm_cp": norm_cp.tolist(),
        "norm_cp_err": norm_cp_err.tolist(),
        "total_seconds": total_s,
    }
    out_path = "results_5x5.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    pr(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
