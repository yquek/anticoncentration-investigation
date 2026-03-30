# Exact Anticoncentration Investigation

This repo contains the exact collision-probability calculation for random geometrically local Hamiltonian time evolution on 2D square lattices.

The main script is `anticoncentration_investigation.py`. It computes

- `CP(t) = sum_x p_x(t)^2`
- the normalized quantity `d * <CP(t)>`

for

- `|psi(t)> = exp(-i H t) |0^n>`
- `H = (1 / sqrt(|P_k|)) sum_{P in P_k} g_P P`
- `g_P ~ N(0, 1)` i.i.d.

where `P_k` is the set of geometrically local Pauli terms on the chosen 2D square lattice.

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`

Install with:

```bash
python -m pip install -r requirements.txt
```

## Basic Run

Example:

```bash
python anticoncentration_investigation.py \
  --shapes 2x2 3x3 4x4 \
  --sample-schedule 2x2:200,3x3:100,4x4:20 \
  --t-max 20 \
  --num-times 80 \
  --out anticoncentration_2d_squares_t20_cp_only.png \
  --log-file anticoncentration_2d_squares_t20_cp_only.json
```

This produces:

- a PNG plot of `d * <CP(t)>`
- a JSON log with the time grid, per-lattice metadata, and the collision-probability curves

## How To Change The Problem Size

The main knobs are:

- `--shapes`
  Example: `--shapes 2x2 3x3 4x4`
- `--sample-schedule`
  Per-lattice sample counts, for example:
  `--sample-schedule 2x2:200,3x3:100,4x4:20`
- `--samples`
  Uniform sample count for all listed shapes. This overrides `--sample-schedule`.
- `--t-max`
  Largest simulated time.
- `--num-times`
  Number of time points between `0` and `t-max`.
- `--k`
  Locality parameter. The current default is `k = 2`.

Examples:

Use a coarser time grid:

```bash
python anticoncentration_investigation.py \
  --shapes 3x3 4x4 \
  --sample-schedule 3x3:100,4x4:20 \
  --t-max 20 \
  --num-times 21
```

Run a single lattice with a uniform sample count:

```bash
python anticoncentration_investigation.py \
  --shapes 4x4 \
  --samples 10 \
  --t-max 10 \
  --num-times 41
```

Try a larger lattice:

```bash
python anticoncentration_investigation.py \
  --shapes 5x5 \
  --samples 1 \
  --t-max 5 \
  --num-times 6
```

## Important Note On Scaling

This implementation is exact. It uses:

- explicit Pauli precomputation over the full computational basis
- sparse Hamiltonian construction
- Krylov `expm_multiply` time evolution of the full statevector

So memory and runtime scale very badly with system size. Large instances such as `5x5` and especially `6x6` may require substantial cluster memory or may still be impractical with this implementation, even on a cluster.

## Outputs

The JSON log contains, for each requested shape:

- lattice label and shape
- `n`, `d`, and `|P_k|`
- number of Hamiltonian samples
- precomputation time
- sampling/runtime information
- `norm_cp`
- `norm_cp_err`

The plot uses a fixed Haar reference line at `2`.
