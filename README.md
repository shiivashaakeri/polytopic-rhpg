# Receding-Horizon Policy Gradient for Polytopic Controller Synthesis

Code for the IEEE CDC 2026 paper *Receding-Horizon Policy Gradient for Polytopic Controller
Synthesis* by Shiva Shakeri, Péter Baranyi, and Mehran Mesbahi
([arXiv:2603.29283](https://arxiv.org/abs/2603.29283)).

The Polytopic Receding-Horizon Policy Gradient (P-RHPG) algorithm synthesizes Parallel
Distributed Compensation (PDC) controllers for polytopic qLPV systems in tensor product (TP)
form. Instead of a common-Lyapunov LMI, P-RHPG minimizes a finite-horizon integrated cost
through a backward-stage decomposition. Each stage subproblem is a strongly convex quadratic
in the vertex gains, so it has a unique global minimizer and gradient descent converges
linearly.

## Repository layout

```
config.py                     experiment settings (single source of truth)
systems/aeroelastic_wing.py   2-DoF aeroelastic wing benchmark (Takarics and Vanek, 2021)
systems/inverted_pendulum.py  additional test system
tp_transform/                 sampling, HOSVD, convex hull, TP model
algorithms/prhpg.py           P-RHPG (Algorithm 1)
algorithms/lmi_synthesis.py   LMI-PDC baseline
algorithms/riccati.py         Riccati utilities
algorithms/quadrature.py      quadrature over the scheduling domain
evaluation/                   frozen-parameter cost, joint stability, metrics, nonlinear simulation
plotting/                     figure helpers
run_experiment_*.py           experiments (see below)
```

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Reproducing the paper

Run the scripts from the repository root, for example:

```bash
python run_experiment_2b.py
```

Results are written to `results/` and figures to `figures/`.

| Result in the paper | Script |
|-|-|
| Fig. 1: optimal cost vs. horizon N for six terminal costs | `run_experiment_2b.py` |
| Fig. 2: closed-loop trajectories at V = 35 m/s | `run_experiment_3.py` (nominal LQR baseline from `run_experiment_5.py`) |
| Table I: P-RHPG vs. LMI-PDC and a nominal LQR gain across vertex grids | `run_experiment_3.py`, `run_experiment_5.py` |
| Fig. 3: gradient norm vs. iteration in the stage solves | `run_experiment_4.py` |

### Additional experiments

- `run_experiment_1.py`: open-loop and closed-loop spectral radius checks and a cost slice.
- `run_experiment_2.py`: horizon sweeps with zero and ARE terminal costs
  (replot with `plot_fig2_from_npz.py`).
- `run_experiment_complexity.py`: direct vs. gradient-descent stage solves across vertex grids.

## Citation

```bibtex
@inproceedings{shakeri2026prhpg,
  title     = {Receding-Horizon Policy Gradient for Polytopic Controller Synthesis},
  author    = {Shakeri, Shiva and Baranyi, P{\'e}ter and Mesbahi, Mehran},
  booktitle = {Proceedings of the IEEE Conference on Decision and Control (CDC)},
  year      = {2026},
  note      = {arXiv:2603.29283}
}
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
