#!/usr/bin/env python3
"""
Anticoncentration of random geometrically-local Hamiltonian time evolution
on 2D square and 3D cubic lattices (OBC).

Ensemble E(k):  H(g) = (1/sqrt(|P_k|)) sum_{P in P_k} g_P P
                g_P ~ N(0,1) iid
P_k = k-local nearest-neighbour Paulis on the lattice.

Uses Krylov-based expm_multiply (not full diagonalisation) for speed.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

_TMP_ROOT = Path(tempfile.gettempdir()) / "analog-quantum-advantage-cache"
_CACHE_DIR = _TMP_ROOT / "xdg-cache"
_MPL_DIR = _TMP_ROOT / "matplotlib"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_CACHE_DIR)
os.environ["MPLCONFIGDIR"] = str(_MPL_DIR)

import numpy as np
import numba
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply, LinearOperator, onenormest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
import time as timer


def pr(*a, **kw):
    print(*a, **kw, flush=True)


def parse_shape(shape):
    x_str, y_str = shape.lower().split("x", 1)
    return int(x_str), int(y_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Investigate anticoncentration for random geometrically-local Hamiltonians."
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["2x2", "3x3", "4x4"],
        help="2D square-lattice shapes to run, formatted as NxM.",
    )
    parser.add_argument("--k", type=int, default=2, help="Locality parameter.")
    parser.add_argument("--t-max", type=float, default=20.0, help="Largest time value.")
    parser.add_argument("--num-times", type=int, default=80, help="Number of time points.")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Uniform number of samples for every lattice. Overrides --sample-schedule when set.",
    )
    parser.add_argument(
        "--sample-schedule",
        type=str,
        default="2x2:200,3x3:100,4x4:20",
        help="Comma-separated per-lattice sample counts, e.g. 2x2:200,3x3:100.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("anticoncentration_2d_squares_t20.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("anticoncentration_2d_squares_t20.json"),
        help="Output JSON log path.",
    )
    return parser.parse_args()


def parse_sample_schedule(schedule):
    result = {}
    if not schedule:
        return result
    for item in schedule.split(","):
        label, value = item.split(":", 1)
        result[label.strip()] = int(value)
    return result


# ── Lattice helpers ──────────────────────────────────────────────────

def square_lattice(nx, ny):
    n = nx * ny
    adj = [[] for _ in range(n)]
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            if x + 1 < nx:
                j = (x + 1) * ny + y
                adj[i].append(j); adj[j].append(i)
            if y + 1 < ny:
                j = x * ny + (y + 1)
                adj[i].append(j); adj[j].append(i)
    return n, adj


def cubic_lattice(nx, ny, nz):
    n = nx * ny * nz
    adj = [[] for _ in range(n)]
    def idx(x, y, z):
        return x * ny * nz + y * nz + z
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                i = idx(x, y, z)
                for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    x2, y2, z2 = x + dx, y + dy, z + dz
                    if x2 < nx and y2 < ny and z2 < nz:
                        j = idx(x2, y2, z2)
                        adj[i].append(j); adj[j].append(i)
    return n, adj


# ── Connected-subgraph enumeration ───────────────────────────────────

def enumerate_connected_subsets(n_sites, adj, max_k):
    result, seen = [], set()
    def grow(fs, ss):
        if fs in seen:
            return
        seen.add(fs)
        result.append(sorted(ss))
        if len(ss) >= max_k:
            return
        for v in ss:
            for u in adj[v]:
                if u not in ss:
                    grow(frozenset(ss | {u}), ss | {u})
    for v in range(n_sites):
        grow(frozenset([v]), {v})
    return result


# ── Precompute Pauli permutations and phases ─────────────────────────

def precompute_paulis(n_sites, adj, k):
    """
    Returns (perms, phases, rows_flat, cols_flat, num_paulis)
    perms[i]  :  destination basis state for Pauli i
    phases[i] :  phase factor for Pauli i
    rows_flat / cols_flat :  precomputed sparse indices
    """
    d = 1 << n_sites
    b = np.arange(d)

    subsets = enumerate_connected_subsets(n_sites, adj, k)
    specs = []
    for subset in subsets:
        for pidx in product(range(3), repeat=len(subset)):
            specs.append(list(zip(subset, pidx)))
    num_paulis = len(specs)

    perms  = np.empty((num_paulis, d), dtype=np.int64)
    phases = np.empty((num_paulis, d), dtype=complex)

    for idx, spec in enumerate(specs):
        flip_mask = 0
        for qubit, pauli in spec:
            if pauli <= 1:
                flip_mask |= 1 << (n_sites - 1 - qubit)
        perms[idx] = b ^ flip_mask
        ph = np.ones(d, dtype=complex)
        for qubit, pauli in spec:
            bit = (b >> (n_sites - 1 - qubit)) & 1
            s = 1 - 2 * bit
            if pauli == 1:
                ph *= 1j * s
            elif pauli == 2:
                ph *= s
        phases[idx] = ph

    rows_flat = perms.ravel()
    cols_flat = np.tile(b, num_paulis)
    return phases, rows_flat, cols_flat, num_paulis


def precompute_pauli_specs(n_sites, adj, k):
    """Compact Pauli specs for on-the-fly matvec — O(num_paulis) memory."""
    subsets = enumerate_connected_subsets(n_sites, adj, k)
    specs = []
    for subset in subsets:
        for pidx in product(range(3), repeat=len(subset)):
            specs.append(list(zip(subset, pidx)))

    num_paulis = len(specs)
    flip_masks = np.empty(num_paulis, dtype=np.int64)
    sign_masks = np.empty(num_paulis, dtype=np.int64)
    base_phases = np.empty(num_paulis, dtype=complex)

    for idx, spec in enumerate(specs):
        fm, sm, num_y = 0, 0, 0
        for qubit, pauli in spec:
            bit_pos = n_sites - 1 - qubit
            if pauli <= 1:  # X or Y → flip
                fm |= 1 << bit_pos
            if pauli >= 1:  # Y or Z → sign
                sm |= 1 << bit_pos
            if pauli == 1:  # Y → factor of i
                num_y += 1
        flip_masks[idx] = fm
        sign_masks[idx] = sm
        base_phases[idx] = 1j ** num_y

    return flip_masks, sign_masks, base_phases, num_paulis


# ── Build sparse Hamiltonian ─────────────────────────────────────────

def build_sparse_H(phases, rows_flat, cols_flat, g, d):
    norm = np.sqrt(len(g))
    vals = ((g[:, None] / norm) * phases).ravel()
    return csr_matrix((vals, (rows_flat, cols_flat)), shape=(d, d))


@numba.njit(parallel=True, cache=True)
def _apply_H_numba(v_re, v_im, flip_masks, sign_masks, w_re, w_im, n_sites):
    d = numba.int64(1) << numba.int64(n_sites)
    num_paulis = len(flip_masks)
    out_re = np.zeros(d, dtype=np.float64)
    out_im = np.zeros(d, dtype=np.float64)
    for b in numba.prange(d):
        re = 0.0
        im = 0.0
        for p in range(num_paulis):
            b_src = b ^ flip_masks[p]
            x = b_src & sign_masks[p]
            x ^= x >> 16
            x ^= x >> 8
            x ^= x >> 4
            x ^= x >> 2
            x ^= x >> 1
            sign = 1 - 2 * (x & 1)
            vr = v_re[b_src]
            vi = v_im[b_src]
            wr = w_re[p]
            wi = w_im[p]
            re += sign * (wr * vr - wi * vi)
            im += sign * (wr * vi + wi * vr)
        out_re[b] = re
        out_im[b] = im
    return out_re, out_im


def build_hamiltonian_linop(flip_masks, sign_masks, base_phases, g, n_sites):
    """LinearOperator for H via on-the-fly matvec (numba-accelerated)."""
    d = 1 << n_sites
    norm = np.sqrt(len(g))
    weighted = (g / norm) * base_phases
    w_re = np.ascontiguousarray(weighted.real)
    w_im = np.ascontiguousarray(weighted.imag)
    fm = np.ascontiguousarray(flip_masks)
    sm = np.ascontiguousarray(sign_masks)

    def _apply_H(v):
        ore, oim = _apply_H_numba(v.real.copy(), v.imag.copy(), fm, sm, w_re, w_im, n_sites)
        return ore + 1j * oim

    def matvec(v):
        return _apply_H(v)

    def rmatvec(v):
        return np.conj(_apply_H(np.conj(v)))

    return LinearOperator((d, d), matvec=matvec, rmatvec=rmatvec, dtype=complex)


def krylov_expm_multiply(H_matvec, v, t, m=30):
    """Compute exp(-i*t*H) @ v using Lanczos iteration for Hermitian H.

    Runs m Lanczos steps on H to build a real symmetric tridiagonal
    approximation T, then computes exp(-i*t*T) on the small matrix exactly.

    Args:
        H_matvec: callable, computes H @ x  (H must be Hermitian)
        v: starting vector (1-D complex array)
        t: scalar time parameter
        m: number of Lanczos steps (controls accuracy)

    Returns:
        Approximation to exp(-i*t*H) @ v
    """
    from scipy.linalg import expm as _dense_expm

    n = len(v)
    m = min(m, n)

    V = np.zeros((m + 1, n), dtype=complex)
    alpha = np.zeros(m, dtype=float)
    beta = np.zeros(m, dtype=float)

    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.zeros_like(v)
    V[0] = v / norm_v

    actual_m = m
    for j in range(m):
        w = H_matvec(V[j])
        alpha[j] = np.real(np.vdot(V[j], w))
        w -= alpha[j] * V[j]
        if j > 0:
            w -= beta[j - 1] * V[j - 1]
        # Re-orthogonalise for numerical stability
        for i in range(j + 1):
            coeff = np.vdot(V[i], w)
            w -= coeff * V[i]
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-14:
            actual_m = j + 1
            break
        V[j + 1] = w / beta[j]
    else:
        actual_m = m

    # Build real symmetric tridiagonal and exponentiate with -i*t
    T = np.diag(alpha[:actual_m])
    if actual_m > 1:
        T += np.diag(beta[:actual_m - 1], 1) + np.diag(beta[:actual_m - 1], -1)
    eT = _dense_expm(-1j * t * T)

    return norm_v * (V[:actual_m].T @ eT[:, 0])


# ── Main experiment loop ─────────────────────────────────────────────

def run_experiment(label, n_sites, adj, k, times, num_samples, rng,
                   lanczos_m=30):
    d = 1 << n_sites
    T = len(times)
    t_start, t_stop = times[0], times[-1]

    pr(f"  {label}: n={n_sites}, d={d}")

    t_pre = timer.time()

    # Estimate memory for dense precompute to choose approach
    subsets = enumerate_connected_subsets(n_sites, adj, k)
    num_paulis_est = sum(3 ** len(s) for s in subsets)
    dense_bytes = num_paulis_est * d * 40
    use_dense = dense_bytes < 2_000_000_000  # 2 GB threshold

    if use_dense:
        phases, rows_flat, cols_flat, num_paulis = precompute_paulis(
            n_sites, adj, k
        )
        precomp_s = timer.time() - t_pre
        pr(f"    |P_k|={num_paulis},  precomp {precomp_s:.2f}s  [sparse]")
    else:
        flip_masks, sign_masks, base_phases, num_paulis = (
            precompute_pauli_specs(n_sites, adj, k)
        )
        precomp_s = timer.time() - t_pre
        pr(f"    |P_k|={num_paulis},  precomp {precomp_s:.2f}s  [on-the-fly]")
        pr(f"    (dense precompute would need {dense_bytes / 1e9:.1f} GB)")

    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0

    cp_all = np.zeros((num_samples, T))

    t0 = timer.time()
    for s in range(num_samples):
        g = rng.standard_normal(num_paulis)

        if use_dense:
            H = build_sparse_H(phases, rows_flat, cols_flat, g, d)
            psi_all = expm_multiply(
                -1j * H, psi0,
                start=t_start, stop=t_stop, num=T, endpoint=True,
            )
            probs = np.abs(psi_all) ** 2
            cp_all[s] = np.sum(probs ** 2, axis=1)
        else:
            H_op = build_hamiltonian_linop(
                flip_masks, sign_masks, base_phases, g, n_sites,
            )
            psi = psi0.copy()
            for t_idx in range(T):
                t_now = times[t_idx]
                t_prev = times[t_idx - 1] if t_idx > 0 else 0.0
                dt = t_now - t_prev
                if dt > 0:
                    psi = krylov_expm_multiply(
                        H_op.matvec, psi, dt, m=lanczos_m,
                    )
                probs = np.abs(psi) ** 2
                cp_all[s, t_idx] = np.sum(probs ** 2)

        if (s + 1) % max(1, num_samples // 10) == 0:
            pr(f"    {s + 1}/{num_samples}  ({timer.time() - t0:.1f}s)")

    sample_s = timer.time() - t0
    norm_cp     = d * np.mean(cp_all, axis=0)
    norm_cp_err = d * np.std(cp_all, axis=0) / np.sqrt(num_samples)
    return {
        "norm_cp": norm_cp,
        "norm_cp_err": norm_cp_err,
        "np_k": num_paulis,
        "precomp_s": precomp_s,
        "sample_s": sample_s,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    times = np.linspace(0, args.t_max, args.num_times)
    schedule = parse_sample_schedule(args.sample_schedule)

    configs = []
    for shape in args.shapes:
        nx, ny = parse_shape(shape)
        n, adj = square_lattice(nx, ny)
        num_samples = args.samples if args.samples is not None else schedule.get(shape)
        if num_samples is None:
            raise ValueError(f"no sample count configured for shape {shape}")
        configs.append((f"2D {nx}x{ny}", n, adj, num_samples, shape))

    results = {}
    for label, n_sites, adj, ns, shape in configs:
        pr(f"\n{'=' * 50}\n  {label}\n{'=' * 50}")
        t0 = timer.time()
        out = run_experiment(label, n_sites, adj, args.k, times, ns, rng)
        pr(f"  DONE  {timer.time() - t0:.1f}s total")
        out.update({"d": 1 << n_sites, "n": n_sites, "samples": ns, "shape": shape})
        results[label] = out

    # ═══════════════════════ PLOTTING ═══════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    labels_2d = [label for label, *_ in configs]

    def palette(labels, cmap_name, lo=0.35, hi=0.85):
        cm = plt.colormaps[cmap_name]
        m = max(len(labels), 1)
        return {l: cm(lo + (hi - lo) * i / max(m - 1, 1))
                for i, l in enumerate(labels)}

    cmap = palette(labels_2d, "viridis")

    for label in labels_2d:
        r = results[label]
        ax.plot(times, r["norm_cp"], color=cmap[label], lw=1.5,
                label=f'{label}  (n={r["n"]}, samples={r["samples"]})')
        ax.fill_between(times,
                        r["norm_cp"] - r["norm_cp_err"],
                        r["norm_cp"] + r["norm_cp_err"],
                        color=cmap[label], alpha=0.12)
    ax.axhline(2, color="black", ls="--", lw=1.2, alpha=0.7, label="Haar (= 2)")
    ax.set_xlabel("Time  t")
    ax.set_ylabel(r"$d\;\langle\mathrm{CP}(t)\rangle$")
    ax.set_title("(a)  Normalised collision probability")
    ax.set_yscale("log")
    ax.set_ylim(1.5, None)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Anticoncentration of $\mathcal{{E}}(k\!=\!{args.k})$ on 2-D square lattices (OBC)",
        fontsize=15, y=1.01)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out
    plt.savefig(out, dpi=150, bbox_inches="tight")
    pr(f"\nFigure saved to {out}")

    log_data = {
        "seed": args.seed,
        "k": args.k,
        "times": times.tolist(),
        "configs": [
            {
                "label": label,
                "shape": results[label]["shape"],
                "n": results[label]["n"],
                "d": results[label]["d"],
                "samples": results[label]["samples"],
                "num_paulis": results[label]["np_k"],
                "precomp_s": results[label]["precomp_s"],
                "sample_s": results[label]["sample_s"],
                "norm_cp": results[label]["norm_cp"].tolist(),
                "norm_cp_err": results[label]["norm_cp_err"].tolist(),
            }
            for label in labels_2d
        ],
        "output_plot": str(out),
    }
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as fh:
        json.dump(log_data, fh, indent=2)
    pr(f"Log saved to {args.log_file}")

    # ═══════════════════════ SUMMARY ═══════════════════════════════
    pr("\n" + "=" * 84)
    pr(f"SUMMARY  (values at t = {times[-1]:.1f})")
    pr("=" * 84)
    pr(f"{'Lattice':<14s} {'n':>3s} {'d':>6s} {'samples':>7s} {'|P_k|':>6s}   "
       f"{'d<CP>':>10s} {'Haar':>8s}")
    pr("-" * 84)
    for label in labels_2d:
        r = results[label]
        haar = 2.0
        pr(f"{label:<14s} {r['n']:3d} {r['d']:6d} {r['samples']:7d} {r['np_k']:6d}   "
           f"{r['norm_cp'][-1]:10.4f} {haar:8.4f}")


if __name__ == "__main__":
    main()
