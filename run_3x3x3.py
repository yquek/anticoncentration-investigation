"""Run exact collision-probability simulation for a 3x3x3 cubic lattice."""

import json
import time as timer

import numpy as np

from anticoncentration_investigation import (
    cubic_lattice,
    precompute_pauli_specs,
    build_hamiltonian_linop,
    krylov_expm_multiply,
)


def pr(*a, **kw):
    print(*a, **kw, flush=True)


def main():
    nx, ny, nz = 3, 3, 3
    k = 2
    seed = 42
    lanczos_m = 15
    num_samples = 4
    times = np.arange(0, 21, dtype=float)  # t = 0, 1, ..., 20

    n_sites, adj = cubic_lattice(nx, ny, nz)
    d = 1 << n_sites
    T = len(times)
    rng = np.random.default_rng(seed)

    pr(f"Lattice: {nx}x{ny}x{nz}, n={n_sites}, d={d:,}")
    pr(f"Times: {times[0]:.0f} to {times[-1]:.0f} ({T} points)")
    pr(f"Lanczos m={lanczos_m}, samples={num_samples}")

    pr("\nPrecomputing Pauli specs...")
    t0 = timer.time()
    flip_masks, sign_masks, base_phases, num_paulis = precompute_pauli_specs(
        n_sites, adj, k
    )
    pr(f"  |P_k| = {num_paulis}, took {timer.time() - t0:.2f}s")

    # Warm up numba JIT on a small problem so cluster jobs don't pay this cost mid-run.
    pr("\nWarming up numba JIT...")
    from anticoncentration_investigation import cubic_lattice as _cl
    _n, _adj = _cl(2, 2, 2)
    _fm, _sm, _bp, _np = precompute_pauli_specs(_n, _adj, 2)
    _g = np.random.default_rng(0).standard_normal(_np)
    _H = build_hamiltonian_linop(_fm, _sm, _bp, _g, _n)
    _H.matvec(np.ones(1 << _n, dtype=complex))
    pr("  JIT compiled.")

    pr("\nBenchmarking matvec on 3x3x3...")
    g_bench = rng.standard_normal(num_paulis)
    H_bench = build_hamiltonian_linop(
        flip_masks, sign_masks, base_phases, g_bench, n_sites
    )
    v_bench = np.zeros(d, dtype=complex)
    v_bench[0] = 1.0
    t0 = timer.time()
    _ = H_bench.matvec(v_bench)
    matvec_s = timer.time() - t0
    pr(f"  Single matvec: {matvec_s:.2f}s")

    # Estimate: each Lanczos step = 1 matvec + reorthog (~30% overhead).
    step_s = matvec_s * 1.3
    time_step_s = step_s * lanczos_m
    sample_s_est = time_step_s * (T - 1)
    total_s_est = sample_s_est * num_samples
    pr(f"\n  === TIMING ESTIMATES ===")
    pr(f"  Per Lanczos step:  ~{step_s:.1f}s")
    pr(f"  Per time step:     ~{time_step_s:.0f}s ({time_step_s/60:.1f} min)")
    pr(f"  Per sample ({T-1} evolve steps): ~{sample_s_est:.0f}s ({sample_s_est/60:.1f} min)")
    pr(f"  Total ({num_samples} samples):  ~{total_s_est:.0f}s ({total_s_est/3600:.1f} hrs)")
    pr(f"  =========================\n")

    # Reset RNG for reproducibility (benchmark consumed one draw).
    rng = np.random.default_rng(seed)

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
            t_step = timer.time()
            if dt > 0:
                psi = krylov_expm_multiply(H_op.matvec, psi, dt, m=lanczos_m)
            probs = np.abs(psi) ** 2
            cp_all[s, t_idx] = np.sum(probs ** 2)
            norm = np.linalg.norm(psi)
            elapsed_total = timer.time() - t_total
            steps_done = s * (T - 1) + max(t_idx, 0)
            steps_total = num_samples * (T - 1)
            eta = (
                elapsed_total / steps_done * (steps_total - steps_done)
                if steps_done > 0
                else total_s_est
            )
            pr(
                f"  sample {s+1}/{num_samples}  t={t_now:2.0f}  "
                f"d*CP={d * cp_all[s, t_idx]:10.4f}  |psi|={norm:.10f}  "
                f"step={timer.time()-t_step:.1f}s  "
                f"elapsed={elapsed_total:.0f}s  ETA={eta:.0f}s ({eta/60:.1f}min)"
            )

        pr(f"  --- Sample {s+1} done in {timer.time() - t_sample:.1f}s ---")

    total_s = timer.time() - t_total
    pr(f"\nAll samples done in {total_s:.1f}s ({total_s/3600:.2f} hrs)")

    norm_cp = d * np.mean(cp_all, axis=0)
    norm_cp_err = d * np.std(cp_all, axis=0) / np.sqrt(num_samples)

    pr(f"\n{'t':>4s}  {'d*CP':>12s}  {'err':>12s}")
    pr("-" * 34)
    for i, t in enumerate(times):
        pr(f"{t:4.0f}  {norm_cp[i]:12.4f}  {norm_cp_err[i]:12.4f}")

    results = {
        "lattice": f"{nx}x{ny}x{nz}",
        "dimension": "3D",
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
    out_path = "results_3x3x3_t20.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    pr(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
