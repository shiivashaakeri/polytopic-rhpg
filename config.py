"""
Experiment Configuration
========================

Single source of truth for all P-RHPG experiments.
All run_experiment_*.py and evaluation modules import from here.

System: 2-DoF Aeroelastic Wing (Takarics & Vanek 2021)
        n=4, m=1, scheduling p=(V, delta_K, delta_C)
"""

import numpy as np
from pathlib import Path

# ── Output paths ─────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── System ────────────────────────────────────────────────────────────────────


def make_system():
    """Return a fresh AeroelasticWing instance (default parameters)."""
    from systems.aeroelastic_wing import AeroelasticWing

    return AeroelasticWing()


# ── TP Model (polytopic approximation) ───────────────────────────────────────

# Fine grid for HOSVD singular-value analysis (not the vertex grid)
TP_SAMPLE_GRID = [20, 5, 5]  # points per scheduling dim for HOSVD
TP_THRESHOLD = 0.01  # relative singular-value cutoff
TP_EXTRA = 1  # vertex grid = HOSVD rank + extra per dim
TP_METHOD = "multilinear"  # recommended for P-RHPG (full-rank Gram)

# Explicit vertex grid sizes to sweep in Experiment 1
# (convergence of approximation quality with N_v)
VERTEX_GRIDS = [
    [2, 2, 2],  # N_v =  8  — coarse
    [3, 2, 2],  # N_v = 12
    [3, 3, 2],  # N_v = 18
    [3, 3, 3],  # N_v = 27  — fine
    [4, 3, 3],  # N_v = 36  — very fine
]

# ── Cost matrices ─────────────────────────────────────────────────────────────
#
# State: x = [h, h_dot, alpha, alpha_dot]
#   h     : plunge displacement   [m]
#   h_dot : plunge rate           [m/s]
#   alpha : pitch angle           [rad]
#   alpha : pitch rate            [rad/s]
#
# Penalise pitch angle most heavily (flutter suppression objective).

Q = np.diag([10.0, 1.0, 100.0, 1.0])  # state penalty  (≻ 0)
R = np.array([[1.0]])  # input penalty  (≻ 0)
Q_N = "are"  # terminal cost: average vertex ARE

# Initial-condition weighting (isotropic — integrates over all ICs equally)
SIGMA0 = np.eye(4)

# ── P-RHPG solver ────────────────────────────────────────────────────────────

# Quadrature grid: 2× vertex density (precomputed once, reused across stages)
QUAD_POINTS = [12, 6, 6]  # per scheduling dimension

# Horizon sweep for Experiment 2 (convergence in N)
HORIZONS = [5, 10, 20, 30, 50, 75, 100, 150, 200]

# Single horizon used for trajectory and comparison experiments
N_HORIZON = 100

# Solver method: 'direct' (exact linear solve) or 'gd' (gradient descent)
SOLVER_METHOD = "direct"

# ── Simulation ────────────────────────────────────────────────────────────────

# Nominal operating point for trajectory plots (below flutter speed)
V_NOM = 20.0  # [m/s]  — airspeed
DK_NOM = 0.0  # nominal stiffness
DC_NOM = 0.0  # nominal damping

# Worst-case operating point for robustness check (above flutter speed)
V_FLUTTER = 30.0  # [m/s]

# Initial condition: small pitch perturbation (realistic IC for flutter test)
X0 = np.array([0.0, 0.0, 0.05, 0.0])  # [h, h_dot, alpha, alpha_dot]

# Closed-loop simulation length
T_SIM = 500  # steps  (= 500 * 0.002 s = 1.0 s)

# ── Experiment 1: Cross-term stability ───────────────────────────────────────
# Compares rho(G_{ii}) (vertex-only) vs rho(Acl(p; K_bar)) (closed-loop)
# over a fine evaluation grid — motivates the PDC cross-term structure.

EXP1_EVAL_GRID = [30, 8, 8]  # fine grid for spectral radius sweep

# ── Experiment 2: Convergence in N ───────────────────────────────────────────
# Plots J_N* vs N — validates Theorem 4.1 (monotone, bounded, convergent).
# Uses the default HORIZONS list above.

# ── Experiment 3: Closed-loop trajectory ─────────────────────────────────────
# State trajectory under P-RHPG vs open-loop (unstable above flutter speed).

EXP3_V = V_FLUTTER  # airspeed above flutter
EXP3_T_SIM = T_SIM

# ── Table: P-RHPG vs LMI ─────────────────────────────────────────────────────
# Columns: method, N_v, J_inf, solve_time, feasible
# Rows: P-RHPG (various N_v), LMI-PDC (various N_v)

TABLE_VERTEX_GRIDS = VERTEX_GRIDS  # same sweep as Experiment 1

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_SEED = 42

# ── Convenience: build the default TPModel ───────────────────────────────────


def make_tp_model(vertex_grid=None, verbose=False):
    """Build and return a TPModel for the aeroelastic wing.

    Parameters
    ----------
    vertex_grid : list of int, optional
        Explicit grid per dim. If None, uses HOSVD-guided selection
        (TP_SAMPLE_GRID + TP_THRESHOLD + TP_EXTRA).
    verbose : bool

    Returns
    -------
    tp : TPModel
    """
    from tp_transform.tp_model import TPModel

    system = make_system()

    if vertex_grid is None:
        return TPModel.from_system(
            system,
            n_sample=TP_SAMPLE_GRID,
            threshold=TP_THRESHOLD,
            extra=TP_EXTRA,
            method=TP_METHOD,
            verbose=verbose,
        )
    else:
        return TPModel.from_grid(system, vertex_grid)


def make_solver(tp_model):
    """Build and return a PRHPG solver for the given TPModel.

    Parameters
    ----------
    tp_model : TPModel

    Returns
    -------
    solver : PRHPG
    """
    from algorithms.prhpg import PRHPG

    return PRHPG(tp_model, Q, R, SIGMA0, quad_points=QUAD_POINTS)


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Config — Self-check")
    print("=" * 60)

    sys = make_system()
    print(f"\nSystem: n={sys.n}, m={sys.n_u}, n_p={sys.n_p}, Ts={sys.Ts}")
    print(f"Flutter speed (nominal): ", end="")
    Vf = sys.flutter_speed()
    print(f"{Vf:.2f} m/s" if Vf else "none in range")

    print(f"\nCost matrices:")
    print(f"  Q  = diag{np.diag(Q).tolist()}")
    print(f"  R  = {R[0, 0]}")
    print(f"  Σ₀ = I_{sys.n}")

    print(f"\nTP model:")
    print(f"  Sample grid : {TP_SAMPLE_GRID}")
    print(f"  Threshold   : {TP_THRESHOLD}")
    print(f"  Vertex grids: {VERTEX_GRIDS}")

    print(f"\nP-RHPG:")
    print(f"  Quad points : {QUAD_POINTS}")
    print(f"  Horizons    : {HORIZONS}")
    print(f"  N (main)    : {N_HORIZON}")
    print(f"  Method      : {SOLVER_METHOD}")

    print(f"\nSimulation:")
    print(f"  x0     = {X0}")
    print(f"  T_sim  = {T_SIM} steps = {T_SIM * sys.Ts:.2f} s")
    print(f"  V_nom  = {V_NOM} m/s  (below flutter)")
    print(f"  V_flutter experiment = {EXP3_V} m/s  (above flutter)")

    print(f"\nOutput dirs:")
    print(f"  results/  → {OUT_DIR}")
    print(f"  figures/  → {FIG_DIR}")
    print(f"  data/     → {DATA_DIR}")

    print("\nConfig OK ✓")
