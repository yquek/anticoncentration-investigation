"""Generate the same style plot as the original script, using 5x5 results."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results_5x5.json") as f:
    data = json.load(f)

times = np.array(data["times"])
norm_cp = np.array(data["norm_cp"])
norm_cp_err = np.array(data["norm_cp_err"])
k = data["k"]
n = data["n_sites"]
d = data["d"]
samples = data["num_samples"]
label = f'2D {data["lattice"]}'

fig, ax = plt.subplots(figsize=(10, 6))

cm = plt.colormaps["viridis"]
color = cm(0.6)

ax.plot(times, norm_cp, color=color, lw=1.5,
        label=f'{label}  (n={n}, samples={samples})')
ax.fill_between(times,
                norm_cp - norm_cp_err,
                norm_cp + norm_cp_err,
                color=color, alpha=0.12)
ax.axhline(2, color="black", ls="--", lw=1.2, alpha=0.7, label="Haar (= 2)")
ax.set_xlabel("Time  t")
ax.set_ylabel(r"$d\;\langle\mathrm{CP}(t)\rangle$")
ax.set_title("(a)  Normalised collision probability")
ax.set_yscale("log")
ax.set_ylim(1.5, None)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

fig.suptitle(
    rf"Anticoncentration of $\mathcal{{E}}(k\!=\!{k})$ on 2-D square lattices (OBC)",
    fontsize=15, y=1.01)
plt.tight_layout()

out = "anticoncentration_5x5_t20.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure saved to {out}")
