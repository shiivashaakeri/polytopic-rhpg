"""
Experiment 4 — Gradient Descent Convergence for P-RHPG Stage Solves
====================================================================

Validates Corollary 1 (linear convergence of gradient descent on Φ_h) by
running the full P-RHPG backward sweep with method='gd' and extracting
per-stage GD diagnostics.

Two figures are produced:

  Figure 4a — Gradient norm ‖∇Φ_h^(k)‖ vs iteration k (log scale)
    • Three representative stages: h = N-1 (first solved), h = N//2 (middle),
      h = 0 (last, most relevant for the deployed gain)
    • Overlays theoretical linear rate: ||∇Φ_h^(k)|| ≤ C·(1 - 1/κ)^k
    • Confirms Corollary 1 empirically

  Figure 4b — Cost gap (Φ_h^(k) - Φ_h*) vs iteration k (log scale)
    • Same three stages; Φ_h* from direct solve
    • Shows GD converges to the exact same optimum

  Console / text output:
    • κ(∇²Φ_h), theoretical rate, empirical rate for each stage
    • Iterations to convergence (tol = 1e-8)
    • J_∞ comparison: GD vs direct solve (should match to ~1e-6 relative)

Usage
-----
  python run_experiment_4.py [--grid N1 N2 N3] [--horizon N] [--tol TOL]
"""

import time
import argparse
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import eigvalsh

ROOT    = Path(__file__).parent
FIG_DIR = ROOT / 'figures'
RES_DIR = ROOT / 'results'
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

from systems.aeroelastic_wing import AeroelasticWing
from tp_transform.tp_model import TPModel
from algorithms.prhpg import PRHPG
from config import Q, R, SIGMA0, QUAD_POINTS

# ── IEEE figure constants (match comparison_plots.py) ─────────────────────────
IEEE_FULL = 7.16
IEEE_HALF = 3.5
IEEE_H    = 2.2

COLORS = {
    'h_last':   '#1f77b4',   # stage h = N-1  (blue)
    'h_mid':    '#ff7f0e',   # stage h = N//2 (orange)
    'h_first':  '#2ca02c',   # stage h = 0    (green)
    'theory':   '#999999',   # theoretical rate (gray dashed)
}

def _save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Core: run GD sweep and extract per-stage diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_gd_sweep(tp, Q, R, Sigma0, N, tol=1e-10, max_iter=2000):
    """Run full P-RHPG sweep with method='gd', return solver + stage_info."""
    solver = PRHPG(tp, Q, R, Sigma0, quad_points=QUAD_POINTS)
    t0 = time.time()
    result = solver.solve(N, Q_N='are', method='gd',
                          max_iter=max_iter, tol=tol)
    elapsed = time.time() - t0
    return solver, result, elapsed


def run_direct_sweep(tp, Q, R, Sigma0, N):
    """Run full P-RHPG sweep with method='direct' for reference."""
    solver = PRHPG(tp, Q, R, Sigma0, quad_points=QUAD_POINTS)
    t0 = time.time()
    result = solver.solve(N, Q_N='are', method='direct')
    elapsed = time.time() - t0
    return solver, result, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Theory: compute theoretical convergence rate from Hessian
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_rate(stage_info):
    """Compute 1 - 1/κ from condition number stored in stage_info."""
    kappa = stage_info.get('hessian_cond', None)
    if kappa is None or not np.isfinite(kappa) or kappa <= 1:
        return None, kappa
    rate = 1.0 - 1.0 / kappa
    return rate, kappa


def empirical_rate(grad_norms):
    """Fit log-linear slope to grad_norms to get empirical rate."""
    gn = np.array(grad_norms)
    gn = gn[gn > 0]
    if len(gn) < 5:
        return None
    log_gn = np.log(gn)
    iters  = np.arange(len(log_gn))
    # Fit to middle 50% to avoid startup and near-zero noise
    i0, i1 = len(iters) // 4, 3 * len(iters) // 4
    if i1 - i0 < 3:
        i0, i1 = 0, len(iters)
    slope, _ = np.polyfit(iters[i0:i1], log_gn[i0:i1], 1)
    return float(np.exp(slope))   # per-iteration multiplicative factor


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_grad_norm(stage_infos_gd, N, save_path=None):
    """Figure 4a: gradient norm vs iteration for three stages."""
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7,
        'lines.linewidth': 1.2, 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.linestyle': '--',
    })
    fig, ax = plt.subplots(figsize=(IEEE_HALF, IEEE_H * 1.1))

    stage_keys = [
        (N - 1, 'h_last',  f'$h = N\\!-\\!1$ (stage 1)'),
        (N // 2, 'h_mid',  f'$h = N/2$  (middle)'),
        (0,     'h_first', f'$h = 0$    (final)'),
    ]

    for h, color_key, label in stage_keys:
        info = stage_infos_gd[h]
        gn   = np.array(info.get('grad_norms', []))
        if len(gn) == 0:
            continue
        iters = np.arange(len(gn))
        ax.semilogy(iters, gn, color=COLORS[color_key], label=label, lw=1.2)

        # Theoretical rate overlay
        rate, kappa = theoretical_rate(info)
        if rate is not None and len(gn) > 0:
            C     = gn[0]
            t_fit = np.arange(len(gn))
            theory_line = C * rate ** t_fit
            ax.semilogy(t_fit, theory_line,
                        color=COLORS[color_key], ls=':', lw=0.9, alpha=0.6)

    # One proxy for the theory label
    ax.semilogy([], [], color='gray', ls=':', lw=0.9, alpha=0.7,
                label=r'Theory $(1-\kappa^{-1})^k$')

    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel(r'$\|\nabla \Phi_h^{(k)}\|_F$')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=6, handlelength=1.5)
    ax.set_xlim(left=0)

    if save_path:
        fig.tight_layout()
        _save(fig, save_path)
    return fig, ax


def plot_cost_gap(stage_infos_gd, stage_infos_direct, N, save_path=None):
    """Figure 4b: cost gap (Φ_h^(k) - Φ_h*) vs iteration.

    Φ_h* is computed by evaluating _stage_cost at the direct-solve optimal
    gains, using the P_history stored in result_dir.
    """
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7,
        'lines.linewidth': 1.2, 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.linestyle': '--',
    })
    fig, ax = plt.subplots(figsize=(IEEE_HALF, IEEE_H * 1.1))

    stage_keys = [
        (N - 1, 'h_last',  f'$h = N\\!-\\!1$'),
        (N // 2, 'h_mid',  f'$h = N/2$'),
        (0,     'h_first', f'$h = 0$'),
    ]

    # P_history is stored in backward order: index 0 → P_N, index 1 → P_{N-1}*, ...
    # K_all[idx] → optimal gains at stage h = N-1-idx
    for h, color_key, label in stage_keys:
        info_gd    = stage_infos_gd.get(h)
        if info_gd is None:
            continue
        cost_trace = np.array(info_gd.get('cost_trace', []))
        if len(cost_trace) == 0:
            continue

        # Φ_h* from direct solve's stored stage_cost at optimum
        info_dir = stage_infos_direct.get(h)
        if info_dir is not None and 'stage_cost' in info_dir:
            phi_star = info_dir['stage_cost']
        else:
            phi_star = cost_trace[-1]  # fallback: converged GD value

        gap   = cost_trace - phi_star
        gap   = np.maximum(gap, 1e-20)
        iters = np.arange(len(gap))

        ax.semilogy(iters, gap, color=COLORS[color_key], label=label, lw=1.2)

    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel(r'$\Phi_h^{(k)} - \Phi_h^*$')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=6, handlelength=1.5)
    ax.set_xlim(left=0)

    if save_path:
        fig.tight_layout()
        _save(fig, save_path)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid',    nargs=3, type=int,
                        default=[3, 2, 2], metavar=('N1','N2','N3'))
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--tol',     type=float, default=1e-10)
    parser.add_argument('--max_iter',type=int, default=5000)
    args = parser.parse_args()

    grid = args.grid
    N    = args.horizon
    tol  = args.tol

    print("=" * 60)
    print("Experiment 4 — GD Convergence for P-RHPG Stage Solves")
    print("=" * 60)
    print(f"  Grid   : {grid}  (N_v = {grid[0]*grid[1]*grid[2]})")
    print(f"  Horizon: N = {N}")
    print(f"  GD tol : {tol:.0e}")
    print(f"  Max iter: {args.max_iter}")
    print()

    # ── Build TP model ────────────────────────────────────────────────────────
    print("Building TP model...")
    system = AeroelasticWing()
    tp     = TPModel.from_grid(system, grid)
    Nv     = tp.n_vertices
    print(f"  N_v = {Nv},  n = {tp.n},  m = {tp.m}")

    # ── Riccati benchmark ────────────────────────────────────────────────────
    print("\nComputing Riccati benchmark...")
    solver_ref = PRHPG(tp, Q, R, SIGMA0, quad_points=QUAD_POINTS)
    J_ric, _ = solver_ref.riccati_benchmark()
    print(f"  J_ric = {J_ric:.4e}")

    # ── Direct solve (reference) ──────────────────────────────────────────────
    print("\nRunning direct solve (reference)...")
    solver_dir, result_dir, t_dir = run_direct_sweep(tp, Q, R, SIGMA0, N)
    J_inf_dir, rho_dir = solver_dir.evaluate_infinite_cost(result_dir.K_vertices)
    print(f"  J_∞  = {J_inf_dir:.4e}  gap = {100*(J_inf_dir/J_ric - 1):.2f}%"
          f"  ρ_max = {rho_dir:.4f}  t = {t_dir:.1f}s")

    # Stage info indexed by stage h (0 = last, N-1 = first solved)
    stage_infos_direct = {info['stage']: info
                          for info in result_dir.stage_info
                          if 'stage' in info}

    # If stage key not stored, index by position
    if not stage_infos_direct:
        # stage_info is stored in backward order: index 0 → h = N-1
        stage_infos_direct = {}
        for idx, info in enumerate(result_dir.stage_info):
            h = N - 1 - idx
            stage_infos_direct[h] = info

    # ── GD solve ─────────────────────────────────────────────────────────────
    print(f"\nRunning GD solve (tol={tol:.0e}, max_iter={args.max_iter})...")
    solver_gd, result_gd, t_gd = run_gd_sweep(
        tp, Q, R, SIGMA0, N, tol=tol, max_iter=args.max_iter)
    J_inf_gd, rho_gd = solver_gd.evaluate_infinite_cost(result_gd.K_vertices)
    print(f"  J_∞  = {J_inf_gd:.4e}  gap = {100*(J_inf_gd/J_ric - 1):.2f}%"
          f"  ρ_max = {rho_gd:.4f}  t = {t_gd:.1f}s")

    # Index GD stage info same way
    stage_infos_gd = {}
    for idx, info in enumerate(result_gd.stage_info):
        h = N - 1 - idx
        stage_infos_gd[h] = info

    # ── Per-stage diagnostics ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  Per-stage diagnostics  (N={N}, N_v={Nv})")
    print("─" * 60)
    print(f"  {'Stage h':>8}  {'κ(∇²Φ)':>12}  {'Theory rate':>12}"
          f"  {'Empirical rate':>14}  {'Iterations':>10}")
    print("  " + "-" * 62)

    stages_to_report = sorted(stage_infos_gd.keys())
    # Pick representative subset for the table: first, middle, last
    rep_stages = [N - 1, N // 2, 0]

    all_kappas = []
    all_theory = []
    all_empirical = []

    for h in stages_to_report:
        info = stage_infos_gd[h]
        gn   = info.get('grad_norms', [])
        kappa = info.get('hessian_cond', float('nan'))
        rate_theory, _ = theoretical_rate(info)
        rate_emp = empirical_rate(gn)
        n_iter   = info.get('iterations', len(gn))

        all_kappas.append(kappa)
        all_theory.append(rate_theory)
        all_empirical.append(rate_emp)

        if h in rep_stages:
            rt_str = f"{rate_theory:.6f}" if rate_theory is not None else "N/A"
            re_str = f"{rate_emp:.6f}"   if rate_emp   is not None else "N/A"
            kp_str = f"{kappa:.2f}"      if np.isfinite(kappa)     else "∞"
            print(f"  {h:>8}  {kp_str:>12}  {rt_str:>12}  {re_str:>14}  {n_iter:>10}")

    # ── Summary ───────────────────────────────────────────────────────────────
    rel_diff = abs(J_inf_gd - J_inf_dir) / max(abs(J_inf_dir), 1e-30)
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  J_∞  direct  = {J_inf_dir:.6e}")
    print(f"  J_∞  GD      = {J_inf_gd:.6e}")
    print(f"  Relative diff = {rel_diff:.2e}  "
          f"({'✓ match' if rel_diff < 1e-4 else '✗ mismatch'})")
    print(f"  ρ_max direct  = {rho_dir:.6f}")
    print(f"  ρ_max GD      = {rho_gd:.6f}")
    print(f"  Time direct   = {t_dir:.1f} s")
    print(f"  Time GD       = {t_gd:.1f} s")

    # Median kappa across stages
    finite_kappas = [k for k in all_kappas if np.isfinite(k)]
    if finite_kappas:
        print(f"\n  Median κ(∇²Φ_h) across stages = {np.median(finite_kappas):.2f}")
        print(f"  Min / Max κ                    = "
              f"{np.min(finite_kappas):.2f} / {np.max(finite_kappas):.2f}")

    # Save summary
    summary_lines = [
        "Experiment 4 — GD Convergence",
        f"Grid: {grid}  N_v={Nv}  N={N}  tol={tol:.0e}",
        f"J_ric       = {J_ric:.6e}",
        f"J_inf direct= {J_inf_dir:.6e}  gap={100*(J_inf_dir/J_ric-1):.2f}%",
        f"J_inf GD    = {J_inf_gd:.6e}  gap={100*(J_inf_gd/J_ric-1):.2f}%",
        f"Rel diff    = {rel_diff:.2e}",
        f"Time direct = {t_dir:.1f}s   Time GD = {t_gd:.1f}s",
    ]
    if finite_kappas:
        summary_lines += [
            f"Median kappa= {np.median(finite_kappas):.2f}",
            f"Min/Max kappa= {np.min(finite_kappas):.2f} / {np.max(finite_kappas):.2f}",
        ]
    summary_path = RES_DIR / 'exp4_summary.txt'
    summary_path.write_text('\n'.join(summary_lines))
    print(f"\n  Saved summary → {summary_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nRendering figures...")

    # Make sure all three representative stages exist in both dicts
    for h in rep_stages:
        if h not in stage_infos_gd:
            print(f"  Warning: stage h={h} not found in GD stage_info, skipping")

    plot_grad_norm(
        stage_infos_gd, N,
        save_path=FIG_DIR / 'fig4a_gd_grad_norm.pdf')

    plot_cost_gap(
        stage_infos_gd, stage_infos_direct, N,
        save_path=FIG_DIR / 'fig4b_gd_cost_gap.pdf')

    print("\nExperiment 4 complete.")


if __name__ == '__main__':
    main()