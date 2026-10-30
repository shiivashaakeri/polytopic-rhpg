"""
Experiment 2 — Convergence in Horizon N
========================================

Validates Theorem 2 (all three parts) by running two horizon sweeps:

  Sweep A  Q_N = 0   : J_N* is monotonically NON-DECREASING  (Thm 2-i, case 1)
  Sweep B  Q_N = are : J_N* is monotonically NON-INCREASING  (Thm 2-i, case 2)

Both sequences are bounded by J_∞(K^feas) (Thm 2-ii) and converge to
the same limit J̄ (Thm 2-iii), squeezing it from below and above.

J̄ is approximated as the average of the two sequences at N=1000.

Outputs
-------
  figures/fig2_convergence.pdf
  results/exp2_zero.npz          — Q_N=0 sweep
  results/exp2_are.npz           — Q_N=are sweep
  results/exp2_summary.txt

Usage
-----
  python run_experiment_2.py [--grid N1 N2 N3] [--horizons N ...] [--no-plot]
"""

import time, argparse, csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent

from systems.aeroelastic_wing import AeroelasticWing
from tp_transform.tp_model   import TPModel
from algorithms.prhpg         import PRHPG
from evaluation.metrics       import suboptimality_gap
from config                   import (Q, R, SIGMA0, HORIZONS, QUAD_POINTS)

FIG_DIR = ROOT / 'figures'
RES_DIR = ROOT / 'results'
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

DEFAULT_VERTEX_GRID = [3, 3, 3]
DEFAULT_HORIZONS    = HORIZONS + [500, 1000]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build TP model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(vertex_grid):
    print(f'\n{"="*60}')
    print(f'Building TP model  vertex_grid={vertex_grid}')
    print(f'{"="*60}')
    system = AeroelasticWing()
    t0 = time.perf_counter()
    tp = TPModel.from_grid(system, vertex_grid)
    print(f'  n={tp.n}, m={tp.m}, n_p={tp.n_p}, N_v={tp.n_vertices}')
    print(f'  TP model built in {time.perf_counter()-t0:.2f} s')
    return tp


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build solver + reference quantities
# ─────────────────────────────────────────────────────────────────────────────

def build_solver(tp):
    print(f'\n{"─"*60}')
    print(f'Building shared PRHPG solver')
    print(f'{"─"*60}')
    t0     = time.perf_counter()
    solver = PRHPG(tp, Q, R, SIGMA0, quad_points=QUAD_POINTS)
    print(f'  N_q = {solver.N_q}  ({time.perf_counter()-t0:.2f} s)')

    print(f'  Computing Riccati lower bound...')
    t0 = time.perf_counter()
    J_ric, K_ric = solver.riccati_benchmark()
    print(f'  J_ric = {J_ric:.6e}  ({time.perf_counter()-t0:.2f} s)')

    # J_feas: ARE-terminated gains at N=50 as K^feas
    print(f'  Computing J_inf(K^feas) (ARE-solve at N=50)...')
    t0 = time.perf_counter()
    try:
        res_feas = solver.solve(50, Q_N='are', method='direct', verbose=False)
        J_feas, rho_feas = solver.evaluate_infinite_cost(res_feas.K_vertices)
        if not np.isfinite(J_feas):
            print(f'  WARNING: not jointly stabilizing (rho={rho_feas:.4f})')
            J_feas = np.nan
        else:
            print(f'  J_feas = {J_feas:.6e}  rho={rho_feas:.4f}  '
                  f'({time.perf_counter()-t0:.2f} s)')
    except Exception as e:
        print(f'  WARNING: {e}')
        J_feas = np.nan

    return solver, J_ric, J_feas


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Horizon sweep (generic)
# ─────────────────────────────────────────────────────────────────────────────

def run_horizon_sweep(solver, horizons, J_ric, Q_N, label):
    """Run backward sweep for each N.

    Q_N : 'zero' | 'are' | ndarray
        'zero' → shaped zero array (non-decreasing J_N*)
        'are'  → vertex ARE average  (non-increasing J_N*)
    """
    print(f'\n{"─"*60}')
    print(f'Horizon sweep [{label}]: N in {horizons}')
    print(f'{"─"*60}')
    print(f'  {"N":>5}  {"J_N*":>14}  {"J_inf":>14}  {"J_ric":>14}  '
          f'{"Gap%":>7}  {"rho":>7}  {"Time":>7}')
    print(f'  {"─"*5}  {"─"*14}  {"─"*14}  {"─"*14}  '
          f'{"─"*7}  {"─"*7}  {"─"*7}')

    records = []

    for N in horizons:
        t0 = time.perf_counter()

        if isinstance(Q_N, str) and Q_N == 'zero':
            n_state = solver.n
            Q_N_arg = np.zeros((solver.N_q, n_state, n_state))
        else:
            Q_N_arg = Q_N  # 'are' string or matrix

        result  = solver.solve(N, Q_N=Q_N_arg, method='direct', verbose=False)
        t_solve = time.perf_counter() - t0

        J_inf, rho_max = solver.evaluate_infinite_cost(result.K_vertices)
        J_N    = float(result.integrated_cost)
        gap    = suboptimality_gap(J_inf, J_ric) if np.isfinite(J_inf) else np.nan
        stable = np.isfinite(J_inf) and rho_max < 1.0

        J_inf_str = f'{J_inf:>14.6e}' if np.isfinite(J_inf) else f'{"inf":>14}'
        flag = 'ok' if stable else 'X'
        print(f'  {N:>5}  {J_N:>14.6e}  {J_inf_str}  {J_ric:>14.6e}  '
              f'{gap:>7.2f}  {rho_max:>7.4f}  {t_solve:>6.2f}s  {flag}')

        records.append(dict(
            N=N, J_N=J_N, J_inf=float(J_inf), J_ric=float(J_ric),
            gap_pct=float(gap), rho_max=float(rho_max),
            solve_time=float(t_solve), stable=bool(stable),
            K_vertices=result.K_vertices.copy(),
        ))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Monotonicity check
# ─────────────────────────────────────────────────────────────────────────────

def check_monotonicity(records, expected='nondecreasing'):
    J_Ns  = np.array([r['J_N'] for r in records])
    diffs = np.diff(J_Ns)
    if expected == 'nondecreasing':
        ok     = bool(np.all(diffs >= -1e-8))
        n_viol = int(np.sum(diffs < -1e-8))
        print(f'  J_N* non-decreasing: {ok}  ({n_viol} violations)')
    else:
        ok     = bool(np.all(diffs <= 1e-8))
        n_viol = int(np.sum(diffs > 1e-8))
        print(f'  J_N* non-increasing: {ok}  ({n_viol} violations)')
    return ok, n_viol


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Estimate J_bar
# ─────────────────────────────────────────────────────────────────────────────

def estimate_J_bar(records_zero, records_are):
    J_bar_lo = float(records_zero[-1]['J_N'])
    J_bar_hi = float(records_are[-1]['J_N'])
    J_bar    = 0.5 * (J_bar_lo + J_bar_hi)
    print(f'\n  J_bar estimate:')
    print(f'    Q_N=0   lower bound : {J_bar_lo:.6e}')
    print(f'    Q_N=are upper bound : {J_bar_hi:.6e}')
    print(f'    Average J_bar       : {J_bar:.6e}')
    print(f'    Squeeze gap         : {J_bar_hi - J_bar_lo:.3e}  '
          f'({(J_bar_hi - J_bar_lo) / J_bar * 100:.3f}%)')
    return J_bar, J_bar_lo, J_bar_hi


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Save
# ─────────────────────────────────────────────────────────────────────────────

def save_sweep(records, label, J_bar, J_feas):
    path = RES_DIR / f'exp2_{label}.npz'
    np.savez(
        path,
        Ns             = np.array([r['N']         for r in records]),
        J_Ns           = np.array([r['J_N']       for r in records]),
        J_infs         = np.array([r['J_inf']     for r in records]),
        J_rics         = np.array([r['J_ric']     for r in records]),
        gap_pcts       = np.array([r['gap_pct']   for r in records]),
        rho_maxs       = np.array([r['rho_max']   for r in records]),
        solve_times    = np.array([r['solve_time']for r in records]),
        K_vertices_all = np.stack([r['K_vertices']for r in records]),
        J_bar          = np.array(J_bar),
        J_feas         = np.array(J_feas if J_feas is not None else np.nan),
    )
    print(f'  Saved {label} -> {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(records_zero, records_are, J_bar, J_feas=None):
    """Figure 2: both J_N* curves squeezing toward J_bar.

    Blue  (Q_N=0,   rising)  validates Thm 2(i) non-decreasing
    Red   (Q_N=are, falling) validates Thm 2(i) non-increasing
    Black dotted              J_bar limit, Thm 2(iii)
    Purple solid              J_feas upper bound, Thm 2(ii)
    """
    print(f'\n{"─"*60}')
    print('Rendering Figure 2')
    print(f'{"─"*60}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    IEEE_HALF = 3.5
    IEEE_H    = 2.4
    DPI       = 300

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7,
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'axes.grid': True, 'grid.alpha': 0.3,
        'grid.linestyle': '--', 'figure.dpi': DPI,
    })

    fig, ax = plt.subplots(figsize=(IEEE_HALF, IEEE_H))

    def filt(records):
        mask = np.array([r['N'] <= 200 for r in records])
        return (np.array([r['N']   for r in records])[mask],
                np.array([r['J_N'] for r in records])[mask])

    Ns_z, JNs_z = filt(records_zero)
    Ns_a, JNs_a = filt(records_are)

    ax.semilogy(Ns_z, JNs_z, color='#1f77b4', marker='s', ms=4, lw=1.4,
                label=r'$J_N^*$, $Q_N{=}0$ (non-decreasing)', zorder=4)

    ax.semilogy(Ns_a, JNs_a, color='#d62728', marker='o', ms=4, lw=1.4,
                label=r'$J_N^*$, $Q_N{=}P_{\mathrm{are}}$ (non-increasing)',
                zorder=4)

    if J_bar is not None and np.isfinite(J_bar):
        ax.axhline(J_bar, color='k', lw=1.0, ls=':',
                   label=r'$\bar{J}$ (limit, $N{=}1000$)', zorder=3)

    if J_feas is not None and np.isfinite(J_feas):
        ax.axhline(J_feas, color='#9467bd', lw=1.0, ls='-',
                   label=r'$J_\infty(\mathbf{K}^{\mathrm{feas}})$ (Thm.~2-ii)',
                   zorder=3)

    y_lo = JNs_z.min() * 0.7
    y_hi = (J_feas if (J_feas is not None and np.isfinite(J_feas))
            else JNs_a.max()) * 1.12
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(r'Horizon $N$')
    ax.set_ylabel(r'$J_N^*$')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax.legend(loc='center right', framealpha=0.9)

    fig.tight_layout()
    out = FIG_DIR / 'fig2_convergence.pdf'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out}')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Experiment 2: convergence in N')
    p.add_argument('--grid',     nargs='+', type=int, default=DEFAULT_VERTEX_GRID)
    p.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
    p.add_argument('--no-plot',  action='store_true')
    return p.parse_args()


def main():
    args        = parse_args()
    vertex_grid = args.grid
    horizons    = sorted(args.horizons)

    print('\nExperiment 2: Convergence in Horizon N')
    print(f'  vertex_grid = {vertex_grid}')
    print(f'  horizons    = {horizons}')
    print(f'  Sweep A: Q_N=0   (non-decreasing — Thm 2-i case 1)')
    print(f'  Sweep B: Q_N=are (non-increasing — Thm 2-i case 2)')

    t_total = time.perf_counter()

    tp                    = build_model(vertex_grid)
    solver, J_ric, J_feas = build_solver(tp)

    # Sweep A: Q_N = 0
    print(f'\n{"="*60}')
    print('Sweep A: Q_N = 0')
    print(f'{"="*60}')
    records_zero = run_horizon_sweep(solver, horizons, J_ric,
                                     Q_N='zero', label='zero')
    print(f'\nMonotonicity (Q_N=0):')
    check_monotonicity(records_zero, expected='nondecreasing')

    # Sweep B: Q_N = are
    print(f'\n{"="*60}')
    print('Sweep B: Q_N = are')
    print(f'{"="*60}')
    records_are = run_horizon_sweep(solver, horizons, J_ric,
                                    Q_N='are', label='are')
    print(f'\nMonotonicity (Q_N=are):')
    check_monotonicity(records_are, expected='nonincreasing')

    # J_bar estimate
    print(f'\n{"─"*60}')
    print('Estimating J_bar')
    print(f'{"─"*60}')
    J_bar, J_bar_lo, J_bar_hi = estimate_J_bar(records_zero, records_are)

    # Save
    print(f'\n{"─"*60}')
    print('Saving')
    print(f'{"─"*60}')
    save_sweep(records_zero, 'zero', J_bar, J_feas)
    save_sweep(records_are,  'are',  J_bar, J_feas)

    # Figure
    if not args.no_plot:
        make_figure(records_zero, records_are, J_bar, J_feas=J_feas)

    print(f'\nExperiment 2 complete in {time.perf_counter()-t_total:.1f} s')

    return dict(
        tp=tp, solver=solver,
        records_zero=records_zero, records_are=records_are,
        J_ric=J_ric, J_feas=J_feas,
        J_bar=J_bar, J_bar_lo=J_bar_lo, J_bar_hi=J_bar_hi,
    )


if __name__ == '__main__':
    main()