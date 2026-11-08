# Golden-Ratio Grasping — Reproduction Code

Reproduction package for the paper

> **Golden-Ratio Grasping: Parameter-Free Real-Time Force Optimization with
> Certified Active-Set Identification**

The paper solves grasping force optimization (GFO) for multifingered robotic
hands with point contacts with friction — a convex circular-cone program —
through its smooth six-dimensional dual, using Malitsky's adaptive golden-ratio
algorithm (aGRAAL). This repository contains the complete code behind every
figure, table entry, and verification claim in the paper.

## Contents

| File | Description |
|---|---|
| `reproduce_experiments.py` | Self-contained Python script. Regenerates all figures and all numerical table entries, and **verifies every theoretical inequality** of the paper on the computed trajectories (stepsize bounds, energy inequality, ergodic rate, Bregman identity, projection KKT conditions, Jacobian vs. finite differences, explicit Lipschitz/curvature constants, quadratic-growth modulus). Aborts with an error if any check fails. |
| `gfo_solvers.c` | C99 implementation of all four solvers (aGRAAL parameter-free, aGRAAL capped, DMPC, FOPC) with identical arithmetic, best-of-9 wall-time measurement, and exact per-iteration flop instrumentation. |
| `case_data.h` | Problem data for the C implementation (both benchmark grasps, hardcoded). |

Both benchmark instances (Case A: five-finger cylindrical grasp; Case B:
three-finger spherical grasp) are fully hardcoded — no external data files are
needed.

## Requirements

- Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`
- A C99 compiler (e.g. `gcc`) for the compiled benchmark

## Usage

### Python (full verification + figures)

```bash
python3 reproduce_experiments.py
```

Runs all experiments and verification checks, writes the figures
(`fig_caseA_convergence.png`, `fig_caseB_convergence.png`, `fig_scalability.png`),
and prints a summary for every table in the paper. Expected final line:

```
All checks completed.  Overall OK = True
```

Runtime: a few minutes on a single core.

### C (compiled timings and flop counts)

```bash
gcc -O2 -o gfo_solvers gfo_solvers.c -lm
./gfo_solvers
```

Prints cold-start and warm-tracking iteration counts, wall times, and exact
flop counts for all methods, plus per-oracle-evaluation costs for
n_f = 3, …, 30 contacts (also written to `c_periter.dat`, which the Python
script picks up for the scalability figure when present; otherwise it falls
back to Python-side measurements).

## Mapping to the paper

| Paper item | Produced by |
|---|---|
| Table 1 (problem constants) | `reproduce_experiments.py` |
| Table 2 (verification of the theory) | `reproduce_experiments.py` (all checks) |
| Table 3 (cold start, 6 methods) | `reproduce_experiments.py` |
| Table 4 (Lipschitz misestimation) | `reproduce_experiments.py` |
| Table 5 (certificates and explicit constants) | `reproduce_experiments.py` |
| Table 6 (warm-started tracking, Python) | `reproduce_experiments.py` |
| Table 7 (compiled timings) | `gfo_solvers.c` |
| Table 8 (robustness) | `reproduce_experiments.py` |
| Table 9 (φ sensitivity + per-evaluation cost) | `reproduce_experiments.py` + `gfo_solvers.c` |
| Figures 1–2 (convergence, scalability) | `reproduce_experiments.py` |

## License

MIT (see `LICENSE`).
