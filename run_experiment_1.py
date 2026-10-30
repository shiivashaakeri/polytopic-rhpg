"""
Experiment 1 — Cross-term Stability and Cost Surface
=====================================================

Validates Corollary 1 (joint stability via Lyapunov) by showing:

  (a) ρ(A_i) > 1 for all open-loop vertices (open-loop unstable near flutter)
  (b) ρ(G_{ii}) < 1 for all vertex closed-loop matrices (vertex-stable)
  (c) ρ(A_cl(p; K̄)) < 1 for all p ∈ Ω (jointly stable under P-RHPG)
  (d) Cross-term spectral radii ρ(G_{ij}), i≠j — the non-trivial stability
      requirement that vertex-only design can violate

  Supplementary: J_∞(V; K̄) vs J_ric(V) — cost slice along airspeed axis,
  showing where P-RHPG tracks Riccati optimality and where it diverges
  (near flutter boundary).

Outputs
-------
  figures/fig1_cross_term.pdf    — Figure 1 for paper (scatter)
  figures/fig1b_cost_slice.pdf   — supplementary cost-vs-airspeed
  results/exp1_summary.txt       — printed summary saved to file
  results/exp1_stab.npz          — raw stability data (archival)

Usage
-----
  python run_experiment_1.py [--grid N1 N2 N3] [--eval N1 N2 N3] [--horizon N]
"""

import time, argparse
import numpy as np
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Expects PYTHONPATH to include the project root (which must contain a
# tp_transform/ sub-package and the flat module files).
# Example flat layout (all files in one directory + tp_transform/ sub-package):
#
#   project/
#     tp_transform/  __init__.py  sampling.py  hosvd.py  convex_hull.py
#     aeroelastic_wing.py   tp_model.py   prhpg.py
#     joint_stability.py    frozen_param_cost.py   comparison_plots.py
#     config.py   run_experiment_1.py
#
ROOT = Path(__file__).parent

from systems.aeroelastic_wing import AeroelasticWing
from tp_transform.tp_model import TPModel
from tp_transform.sampling import ParameterGrid
from algorithms.prhpg import run_prhpg

from evaluation.joint_stability import stability_comparison
from evaluation.frozen_param_cost import marginal_cost_1d, riccati_cost_at_p
from plotting.comparison_plots import plot_cross_term_stability, plot_cost_slice_1d, format_comparison_table
from config import Q, R, SIGMA0, N_HORIZON, QUAD_POINTS

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_VERTEX_GRID = [3, 2, 2]  # 12 vertices — compact, well-conditioned
DEFAULT_EVAL_GRID = [20, 8, 8]  # 1280 evaluation points (fine enough for joint)
DEFAULT_HORIZON = N_HORIZON  # from config (100)
COST_SLICE_N = 60  # points along V axis for supplementary figure
FLUTTER_SPEED = 25.56  # m/s (nominal, delta_K=delta_C=0)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build system and TP model
# ─────────────────────────────────────────────────────────────────────────────


def build_tp_model(vertex_grid):
    """Build AeroelasticWing + TPModel from vertex grid spec."""
    print(f"\n{'=' * 60}")
    print(f"Building TP model  vertex_grid={vertex_grid}")
    print(f"{'=' * 60}")

    system = AeroelasticWing()
    t0 = time.perf_counter()
    tp = TPModel.from_grid(system, vertex_grid)
    print(f"  System: n={tp.n}, m={tp.m}, n_p={tp.n_p}")
    print(f"  Param bounds:")
    labels = ["V [m/s]", "δK", "δC"]
    for i, (lo, hi) in enumerate(system.param_bounds):
        print(f"    {labels[i]:8s}: [{lo:+.3f}, {hi:+.3f}]")
    print(f"  Flutter speed (nominal): {FLUTTER_SPEED:.2f} m/s")
    print(f"  TP model built in {time.perf_counter() - t0:.2f} s")
    print(f"  N_v = {tp.n_vertices}  (grid {list(tp.snnn.ranks)})")

    return system, tp


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run P-RHPG
# ─────────────────────────────────────────────────────────────────────────────


def run_synthesis(tp, horizon):
    """Run P-RHPG and return (result, solver)."""
    print(f"\n{'─' * 60}")
    print(f"Running P-RHPG  N={horizon}")
    print(f"{'─' * 60}")

    t0 = time.perf_counter()
    result, solver = run_prhpg(
        tp,
        N=horizon,
        Q=Q,
        R=R,
        Sigma0=SIGMA0,
        Q_N="are",
        method="direct",
        quad_points=QUAD_POINTS,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    J_inf, rho_max = solver.evaluate_infinite_cost(result.K_vertices)
    J_ric, _ = solver.riccati_benchmark()

    print(f"\n  Synthesis summary:")
    print(f"    J_N  (finite-horizon)  = {result.integrated_cost:.6e}")
    print(f"    J_∞  (infinite-horizon)= {J_inf:.6e}")
    print(f"    J_ric (Riccati LB)     = {J_ric:.6e}")
    print(f"    Gap = {(J_inf - J_ric) / J_ric * 100:.2f}%")
    print(f"    ρ_max over Ω           = {rho_max:.6f}")
    print(f"    Solve time             = {elapsed:.2f} s")

    return result, solver, J_inf, J_ric, rho_max, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Stability comparison
# ─────────────────────────────────────────────────────────────────────────────


def run_stability_analysis(tp, K_vertices, eval_grid_spec):
    """Compute and print stability_comparison dict."""
    print(f"\n{'─' * 60}")
    print(f"Stability analysis  eval_grid={eval_grid_spec}")
    print(f"{'─' * 60}")

    eval_grid = ParameterGrid(tp.system.param_bounds, n_points=eval_grid_spec)
    N_eval = eval_grid.total_points
    print(f"  Evaluation grid: {eval_grid_spec} = {N_eval} points")

    t0 = time.perf_counter()
    sd = stability_comparison(tp, K_vertices, eval_grid)
    print(f"  Computed in {time.perf_counter() - t0:.2f} s")

    N_v = tp.n_vertices
    print(f"\n  ── Spectral radii ──")
    print(f"  Open-loop vertices  ρ(A_i):")
    for i, r in enumerate(sd["rho_open"]):
        flag = "✗ UNSTABLE" if r >= 1 else "✓"
        print(f"    i={i + 1:2d}: ρ = {r:.4f}  {flag}")

    print(f"\n  Diagonal CL ρ(G_{{ii}}) — vertex stability:")
    for i, r in enumerate(sd["rho_diagonal"]):
        flag = "✓" if r < 1 else "✗ UNSTABLE"
        print(f"    i={i + 1:2d}: ρ = {r:.4f}  {flag}")

    print(f"\n  Cross-term summary:")
    print(f"    max ρ(G_{{ij}}) over ALL i,j : {sd['max_rho_cross']:.4f}")
    print(f"    max ρ(G_{{ij}}) over i≠j     : {sd['max_rho_offdiag']:.4f}")
    print(f"    All vertex-stable             : {sd['all_vertex_stable']}")
    print(f"    All cross-stable              : {sd['all_cross_stable']}")

    print(f"\n  Joint CL ρ(A_cl(p; K̄)) over {N_eval} grid points:")
    print(f"    max ρ                         : {sd['max_rho_joint']:.4f}")
    print(f"    Fraction stable               : {sd['stable_joint'].mean() * 100:.1f}%")
    print(f"    All jointly stable            : {sd['all_joint_stable']}")

    # Warn if not fully stable (should not happen for a well-tuned P-RHPG)
    if not sd["all_joint_stable"]:
        n_unstable = (~sd["stable_joint"]).sum()
        print(f"\n  ⚠ WARNING: {n_unstable} evaluation points are unstable.")
        print(f"    Consider increasing N or using a finer vertex grid.")

    return sd, eval_grid


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Cost slice along V axis
# ─────────────────────────────────────────────────────────────────────────────


def run_cost_slice(tp, K_vertices, n_points=COST_SLICE_N):
    """J_∞(V; K̄) vs J_ric(V) along the airspeed axis."""
    print(f"\n{'─' * 60}")
    print(f"Cost slice along V axis  ({n_points} points)")
    print(f"{'─' * 60}")

    t0 = time.perf_counter()
    p_vals, pdc_costs, rho_vals = marginal_cost_1d(
        tp,
        K_vertices,
        dim=0,  # V is dim 0
        n_points=n_points,
        Q=Q,
        R=R,
        Sigma0=SIGMA0,
    )
    print(f"  PDC costs computed in {time.perf_counter() - t0:.2f} s")

    # Riccati cost at same slice — evaluate pointwise
    t0 = time.perf_counter()
    bounds = tp.system.param_bounds
    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])  # midpoint of δK, δC
    n_p = tp.n_p

    from evaluation.frozen_param_cost import riccati_cost_at_p

    ric_costs = np.full(n_points, np.nan)
    for k, V in enumerate(p_vals):
        p = mid.copy()
        p[0] = V
        S = tp.system.system_matrix(p, discrete=True)
        A_p = S[:, : tp.n]
        B_p = S[:, tp.n :]
        cost, _ = riccati_cost_at_p(A_p, B_p, Q, R, SIGMA0)
        ric_costs[k] = cost
    print(f"  Riccati costs computed in {time.perf_counter() - t0:.2f} s")

    n_stable_pdc = np.isfinite(pdc_costs).sum()
    n_stable_ric = np.isfinite(ric_costs).sum()
    print(f"\n  PDC stable over V range : {n_stable_pdc}/{n_points} points")
    print(f"  Riccati stable          : {n_stable_ric}/{n_points} points")

    # Cost comparison at nominal V=20 (well below flutter)
    k_nom = np.argmin(np.abs(p_vals - 20.0))
    print(f"\n  At V=20 m/s:")
    print(f"    J_∞ (P-RHPG) = {pdc_costs[k_nom]:.4e}")
    print(f"    J_ric        = {ric_costs[k_nom]:.4e}")
    if np.isfinite(pdc_costs[k_nom]) and np.isfinite(ric_costs[k_nom]):
        gap = (pdc_costs[k_nom] - ric_costs[k_nom]) / ric_costs[k_nom] * 100
        print(f"    Gap          = {gap:.2f}%")

    return p_vals, pdc_costs, ric_costs, rho_vals


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Plot and save
# ─────────────────────────────────────────────────────────────────────────────


def make_figures(sd, p_vals, pdc_costs, ric_costs, rho_vals):
    """Render and save all figures for Experiment 1."""
    print(f"\n{'─' * 60}")
    print(f"Rendering figures")
    print(f"{'─' * 60}")

    # Figure 1: cross-term stability scatter
    p_fig1 = FIG_DIR / "fig1_cross_term.pdf"
    plot_cross_term_stability(
        sd,
        save_path=p_fig1,
        title=None,  # no title in IEEE paper figures
    )

    # Figure 1b: cost slice along V (supplementary)
    p_fig1b = FIG_DIR / "fig1b_cost_slice.pdf"
    plot_cost_slice_1d(
        p_vals,
        pdc_costs,
        ric_costs,
        rho_vals=rho_vals,
        xlabel=r"Airspeed $V$ [m/s]",
        flutter_speed=FLUTTER_SPEED,
        save_path=p_fig1b,
        title=None,
    )

    return p_fig1, p_fig1b


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Save raw data and summary
# ─────────────────────────────────────────────────────────────────────────────


def save_results(
    sd,
    K_vertices,
    p_vals,
    pdc_costs,
    ric_costs,
    rho_vals,
    J_inf,
    J_ric,
    rho_max_prhpg,
    elapsed_prhpg,
    vertex_grid,
    horizon,
):
    """Archive raw arrays and write human-readable summary."""

    # NumPy archive
    npz_path = RES_DIR / "exp1_stab.npz"
    np.savez(
        npz_path,
        rho_open=sd["rho_open"],
        rho_diagonal=sd["rho_diagonal"],
        rho_cross=sd["rho_cross"],
        rho_joint=sd["rho_joint"],
        stable_joint=sd["stable_joint"],
        grid_pts=sd["grid_pts"],
        K_vertices=K_vertices,
        p_vals_slice=p_vals,
        pdc_costs_slice=pdc_costs,
        ric_costs_slice=ric_costs,
        rho_vals_slice=rho_vals,
    )
    print(f"\n  Saved raw data → {npz_path}")

    # Human-readable summary
    lines = [
        "=" * 60,
        "Experiment 1 — Cross-term Stability and Cost Surface",
        "=" * 60,
        f"Vertex grid        : {vertex_grid}  (N_v = {sd['rho_open'].size})",
        f"Horizon N          : {horizon}",
        f"Eval grid points   : {sd['rho_joint'].size}",
        "",
        "── P-RHPG Synthesis ─────────────────────────────────────",
        f"J_∞  (infinite)    : {J_inf:.6e}",
        f"J_ric (Riccati LB) : {J_ric:.6e}",
        f"Gap                : {(J_inf - J_ric) / J_ric * 100:.2f}%",
        f"ρ_max over Ω       : {rho_max_prhpg:.6f}",
        f"Solve time         : {elapsed_prhpg:.2f} s",
        "",
        "── Stability ────────────────────────────────────────────",
        f"max ρ(A_i)         : {sd['max_rho_open']:.4f}  (open-loop)",
        f"max ρ(G_{{ii}})      : {sd['max_rho_diagonal']:.4f}  (vertex CL)",
        f"max ρ(G_{{ij}}) i≠j  : {sd['max_rho_offdiag']:.4f}  (cross-term)",
        f"max ρ(Acl(p;K̄))   : {sd['max_rho_joint']:.4f}  (joint CL)",
        f"All vertex-stable  : {sd['all_vertex_stable']}",
        f"All cross-stable   : {sd['all_cross_stable']}",
        f"All jointly stable : {sd['all_joint_stable']}",
        f"Frac stable        : {sd['stable_joint'].mean() * 100:.1f}%",
        "",
        "── Figures ──────────────────────────────────────────────",
        f"fig1_cross_term.pdf",
        f"fig1b_cost_slice.pdf",
        "=" * 60,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)

    txt_path = RES_DIR / "exp1_summary.txt"
    txt_path.write_text(summary)
    print(f"\n  Saved summary → {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1: cross-term stability")
    p.add_argument(
        "--grid", nargs="+", type=int, default=DEFAULT_VERTEX_GRID, help=f"Vertex grid (default: {DEFAULT_VERTEX_GRID})"
    )
    p.add_argument(
        "--eval", nargs="+", type=int, default=DEFAULT_EVAL_GRID, help=f"Evaluation grid (default: {DEFAULT_EVAL_GRID})"
    )
    p.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON, help=f"P-RHPG horizon N (default: {DEFAULT_HORIZON})"
    )
    p.add_argument("--no-slice", action="store_true", help="Skip cost slice computation (faster)")
    return p.parse_args()


def main():
    args = parse_args()

    vertex_grid = args.grid
    eval_grid = args.eval
    horizon = args.horizon

    print("\nExperiment 1: Cross-term Stability and Cost Surface")
    print(f"  vertex_grid = {vertex_grid}")
    print(f"  eval_grid   = {eval_grid}")
    print(f"  horizon N   = {horizon}")

    t_total = time.perf_counter()

    # ── Steps ─────────────────────────────────────────────────────────────────
    system, tp = build_tp_model(vertex_grid)

    result, solver, J_inf, J_ric, rho_max, elapsed = run_synthesis(tp, horizon)

    sd, _ = run_stability_analysis(tp, result.K_vertices, eval_grid)

    if not args.no_slice:
        p_vals, pdc_costs, ric_costs, rho_vals = run_cost_slice(tp, result.K_vertices)
    else:
        print("\n  (Cost slice skipped via --no-slice)")
        bounds = system.param_bounds
        p_vals = np.linspace(bounds[0, 0], bounds[0, 1], 2)
        pdc_costs = ric_costs = rho_vals = np.full(2, np.nan)

    make_figures(sd, p_vals, pdc_costs, ric_costs, rho_vals)

    save_results(
        sd,
        result.K_vertices,
        p_vals,
        pdc_costs,
        ric_costs,
        rho_vals,
        J_inf,
        J_ric,
        rho_max,
        elapsed,
        vertex_grid,
        horizon,
    )

    print(f"\nExperiment 1 complete in {time.perf_counter() - t_total:.1f} s")

    # Return data dict for use from other scripts (e.g. run_experiment_3.py)
    return dict(
        tp=tp,
        K_vertices=result.K_vertices,
        J_inf=J_inf,
        J_ric=J_ric,
        rho_max=rho_max,
        stab_dict=sd,
        p_vals=p_vals,
        pdc_costs=pdc_costs,
        ric_costs=ric_costs,
        rho_vals=rho_vals,
    )


if __name__ == "__main__":
    main()
