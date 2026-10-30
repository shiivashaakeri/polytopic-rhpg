"""
Experiment 2b — Terminal Cost Sensitivity
==========================================

Tests whether J_N* converges to the same limit J_bar for arbitrary Q_N,
not just Q_N=0 or Q_N=P_are.

Candidates:
  A) Q_N = 0           (baseline, non-decreasing)
  B) Q_N = P_are       (baseline, non-increasing)
  C) Q_N = Q           (running cost matrix)
  D) Q_N = 1000*I      (absurdly large)
  E) Q_N = 0.01*I      (very small nonzero)
  F) Q_N = random PSD  (Wishart draw)

If all converge to the same J_bar, the terminal cost washes out
regardless of Q_N. If any diverges or oscillates, the general
claim needs a caveat.

Usage:
  python run_experiment_2b.py [--grid N1 N2 N3] [--horizons N ...]
"""

import time, argparse
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
DEFAULT_HORIZONS    = [5, 10, 20, 30, 50, 75, 100, 150, 200, 500, 1000]


def build_model(vertex_grid):
    print(f'\n{"="*60}')
    print(f'Building TP model  vertex_grid={vertex_grid}')
    print(f'{"="*60}')
    system = AeroelasticWing()
    tp = TPModel.from_grid(system, vertex_grid)
    print(f'  n={tp.n}, m={tp.m}, N_v={tp.n_vertices}')
    return tp


def build_solver(tp):
    print(f'\n{"="*60}')
    print(f'Building PRHPG solver')
    print(f'{"="*60}')
    solver = PRHPG(tp, Q, R, SIGMA0, quad_points=QUAD_POINTS)
    print(f'  N_q = {solver.N_q}')

    J_ric, _ = solver.riccati_benchmark()
    print(f'  J_ric = {J_ric:.6e}')

    # J_feas from ARE-terminated N=50
    try:
        res_feas = solver.solve(50, Q_N='are', method='direct', verbose=False)
        J_feas, rho_feas = solver.evaluate_infinite_cost(res_feas.K_vertices)
        print(f'  J_feas = {J_feas:.6e}  rho={rho_feas:.4f}')
    except Exception as e:
        print(f'  WARNING: {e}')
        J_feas = np.nan

    return solver, J_ric, J_feas


def build_QN_candidates(solver):
    """Build a dict of Q_N candidates to test."""
    n = solver.n
    N_q = solver.N_q

    # Random PSD: Wishart draw (reproducible)
    rng = np.random.default_rng(seed=42)
    W = rng.standard_normal((n, 2*n))
    Q_random = W @ W.T  # rank-2n PSD, entries ~ O(n)

    candidates = {
        'zero':      np.zeros((N_q, n, n)),
        'are':       'are',
        'Q':         np.tile(Q[np.newaxis], (N_q, 1, 1)),
        '1000I':     np.tile((1000.0 * np.eye(n))[np.newaxis], (N_q, 1, 1)),
        '0.01I':     np.tile((0.01 * np.eye(n))[np.newaxis], (N_q, 1, 1)),
        'random':    np.tile(Q_random[np.newaxis], (N_q, 1, 1)),
    }

    print(f'\n{"="*60}')
    print(f'Q_N candidates:')
    print(f'{"="*60}')
    for name, val in candidates.items():
        if isinstance(val, str):
            print(f'  {name:>10s}:  (string "{val}")')
        else:
            eigmax = np.linalg.eigvalsh(val[0]).max()
            eigmin = np.linalg.eigvalsh(val[0]).min()
            print(f'  {name:>10s}:  eig_min={eigmin:.4f}  eig_max={eigmax:.4f}')

    return candidates


def run_sweep(solver, horizons, Q_N_val, label, J_ric):
    """Run P-RHPG for each horizon N, return list of records."""
    print(f'\n{"─"*60}')
    print(f'Sweep [{label}]')
    print(f'{"─"*60}')
    print(f'  {"N":>5}  {"J_N*":>14}  {"J_inf":>14}  {"rho":>7}  {"Time":>7}')
    print(f'  {"─"*5}  {"─"*14}  {"─"*14}  {"─"*7}  {"─"*7}')

    records = []
    for N in horizons:
        t0 = time.perf_counter()
        result = solver.solve(N, Q_N=Q_N_val, method='direct', verbose=False)
        dt = time.perf_counter() - t0

        J_N = float(result.integrated_cost)
        J_inf, rho = solver.evaluate_infinite_cost(result.K_vertices)
        stable = np.isfinite(J_inf) and rho < 1.0

        J_inf_s = f'{J_inf:>14.6e}' if np.isfinite(J_inf) else f'{"inf":>14}'
        flag = 'ok' if stable else 'X'
        print(f'  {N:>5}  {J_N:>14.6e}  {J_inf_s}  {rho:>7.4f}  {dt:>6.2f}s  {flag}')

        records.append(dict(N=N, J_N=J_N, J_inf=float(J_inf),
                            rho_max=float(rho), stable=stable, time=dt))
    return records


def analyze_convergence(all_records, J_ric):
    """Compare final J_N* values across all Q_N choices."""
    print(f'\n{"="*60}')
    print(f'CONVERGENCE ANALYSIS')
    print(f'{"="*60}')

    # Get J_bar estimate from zero and are
    J_zero_final = all_records['zero'][-1]['J_N']
    J_are_final  = all_records['are'][-1]['J_N']
    J_bar = 0.5 * (J_zero_final + J_are_final)

    print(f'\n  Reference values:')
    print(f'    J_bar (avg of zero/are at max N) = {J_bar:.6e}')
    print(f'    J_ric                             = {J_ric:.6e}')
    print(f'    zero  final J_N*                  = {J_zero_final:.6e}')
    print(f'    are   final J_N*                  = {J_are_final:.6e}')
    print(f'    squeeze gap                       = {abs(J_are_final - J_zero_final)/J_bar*100:.4f}%')

    print(f'\n  {"Label":>10s}  {"J_N*(max N)":>14s}  {"Gap to J_bar":>14s}  '
          f'{"Rel gap%":>10s}  {"Monotone?":>10s}  {"Stable?":>10s}')
    print(f'  {"─"*10}  {"─"*14}  {"─"*14}  {"─"*10}  {"─"*10}  {"─"*10}')

    for label, records in all_records.items():
        J_final = records[-1]['J_N']
        gap = J_final - J_bar
        rel = gap / J_bar * 100

        # Check monotonicity
        J_Ns = [r['J_N'] for r in records]
        diffs = np.diff(J_Ns)
        if np.all(diffs >= -1e-6):
            mono = 'non-decr'
        elif np.all(diffs <= 1e-6):
            mono = 'non-incr'
        else:
            mono = 'NEITHER'

        # Check stability at final N
        stable_final = records[-1]['stable']

        print(f'  {label:>10s}  {J_final:>14.6e}  {gap:>+14.3e}  '
              f'{rel:>+10.4f}  {mono:>10s}  {"yes" if stable_final else "NO":>10s}')

    # Detailed: J_N* at each horizon for all candidates
    print(f'\n  J_N* at each horizon:')
    horizons = [r['N'] for r in all_records['zero']]
    header = f'  {"N":>5}'
    for label in all_records:
        header += f'  {label:>14s}'
    print(header)
    print(f'  {"─"*5}' + f'  {"─"*14}' * len(all_records))

    for idx, N in enumerate(horizons):
        row = f'  {N:>5}'
        for label, records in all_records.items():
            if idx < len(records):
                row += f'  {records[idx]["J_N"]:>14.6e}'
            else:
                row += f'  {"---":>14s}'
        print(row)

    return J_bar


def make_figure(all_records, J_bar, J_feas):
    """Plot all J_N* curves on the same axes."""
    print(f'\n{"─"*60}')
    print('Rendering Figure 2b')
    print(f'{"─"*60}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    colors = {
        'zero':   '#1f77b4',
        'are':    '#d62728',
        'Q':      '#2ca02c',
        '1000I':  '#ff7f0e',
        '0.01I':  '#9467bd',
        'random': '#8c564b',
    }
    markers = {
        'zero': 's', 'are': 'o', 'Q': '^',
        '1000I': 'D', '0.01I': 'v', 'random': 'P',
    }
    labels = {
        'zero':   r'$Q_N = 0$',
        'are':    r'$Q_N = P_{\mathrm{are}}$',
        'Q':      r'$Q_N = Q$',
        '1000I':  r'$Q_N = 1000\,I$',
        '0.01I':  r'$Q_N = 0.01\,I$',
        'random': r'$Q_N = \mathrm{random\ PSD}$',
    }

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.grid': True, 'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: full log-log view
    for label, records in all_records.items():
        Ns = [r['N'] for r in records]
        JNs = [r['J_N'] for r in records]
        ax1.semilogy(Ns, JNs, color=colors[label], marker=markers[label],
                     ms=4, lw=1.2, label=labels[label])

    if np.isfinite(J_bar):
        ax1.axhline(J_bar, color='k', lw=1.0, ls=':', label=r'$\bar{J}$')
    if np.isfinite(J_feas):
        ax1.axhline(J_feas, color='purple', lw=0.8, ls='-', alpha=0.5,
                     label=r'$J_\infty(\mathbf{K}^{\mathrm{feas}})$')

    ax1.set_xlabel('Horizon $N$')
    ax1.set_ylabel(r'$J_N^*$')
    ax1.set_title('Full view (log scale)')
    ax1.legend(fontsize=7, loc='center right')

    # Right panel: zoom on convergence region (N >= 50, linear y)
    for label, records in all_records.items():
        mask = [(r['N'] >= 50) for r in records]
        Ns = [r['N'] for r, m in zip(records, mask) if m]
        JNs = [r['J_N'] for r, m in zip(records, mask) if m]
        if Ns:
            ax2.plot(Ns, JNs, color=colors[label], marker=markers[label],
                     ms=4, lw=1.2, label=labels[label])

    if np.isfinite(J_bar):
        ax2.axhline(J_bar, color='k', lw=1.0, ls=':')

    ax2.set_xlabel('Horizon $N$')
    ax2.set_ylabel(r'$J_N^*$')
    ax2.set_title('Zoom: $N \\geq 50$ (linear scale)')
    ax2.set_xscale('log')
    ax2.legend(fontsize=7)

    fig.tight_layout()
    out = FIG_DIR / 'fig2b_terminal_sensitivity.pdf'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', nargs='+', type=int, default=DEFAULT_VERTEX_GRID)
    parser.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    horizons = sorted(args.horizons)
    print('\nExperiment 2b: Terminal Cost Sensitivity')
    print(f'  grid     = {args.grid}')
    print(f'  horizons = {horizons}')

    t_total = time.perf_counter()

    tp = build_model(args.grid)
    solver, J_ric, J_feas = build_solver(tp)
    candidates = build_QN_candidates(solver)

    # Run all sweeps
    all_records = {}
    for label, Q_N_val in candidates.items():
        all_records[label] = run_sweep(solver, horizons, Q_N_val, label, J_ric)

    # Analysis
    J_bar = analyze_convergence(all_records, J_ric)

    # Save
    np.savez(
        RES_DIR / 'exp2b_terminal_sensitivity.npz',
        **{f'{label}_Ns': np.array([r['N'] for r in recs])
           for label, recs in all_records.items()},
        **{f'{label}_JNs': np.array([r['J_N'] for r in recs])
           for label, recs in all_records.items()},
        **{f'{label}_Jinfs': np.array([r['J_inf'] for r in recs])
           for label, recs in all_records.items()},
        **{f'{label}_rhos': np.array([r['rho_max'] for r in recs])
           for label, recs in all_records.items()},
        J_bar=J_bar, J_ric=J_ric, J_feas=J_feas,
    )
    print(f'\n  Saved -> {RES_DIR / "exp2b_terminal_sensitivity.npz"}')

    # Figure
    if not args.no_plot:
        make_figure(all_records, J_bar, J_feas)

    print(f'\nExperiment 2b complete in {time.perf_counter()-t_total:.1f} s')


if __name__ == '__main__':
    main()