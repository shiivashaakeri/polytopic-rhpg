"""
Experiment 3 — Trajectory and Method Comparison
================================================

Two sub-tasks:

(A) Trajectory plot at V = 30 m/s (above flutter speed ~25.56 m/s)
    - Open-loop: uncontrolled, flutters and diverges
    - P-RHPG:    PDC gain from Experiment 2 best-N, schedules on p=(V,δK,δC)
    - LMI-PDC:   synthesised gain at same vertex grid, same cost matrices
    Shows flutter suppression on pitch angle α(t) and plunge h(t).

(B) Comparison table across vertex grids
    For each grid in TABLE_VERTEX_GRIDS:
      - Run P-RHPG (N = N_HORIZON)
      - Run LMI synthesis  (objective = 'h2')
      - Evaluate J_∞ for both using the shared Riccati quadrature
      - Record solve times, feasibility, stability
    Saves both a CSV and a LaTeX tabular fragment.

Note on scheduling for the aeroelastic wing
-------------------------------------------
The wing scheduling parameter p = (V, δK, δC) is external (not a function of
state x).  `tp_model.simulate_nonlinear` requires `system.scheduling_param`,
which the wing does not provide.  All trajectory simulations therefore use
`_simulate_pdc_fixed_p` — a thin, self-contained PDC loop that holds p constant
at the operating point (V=EXP3_V, δK=0, δC=0).

Outputs
-------
  figures/fig3_trajectory.pdf      — Figure 3 for paper
  results/exp3_table.csv           — comparison table (CSV)
  results/exp3_table.tex           — comparison table (LaTeX fragment)
  results/exp3_summary.txt         — human-readable summary

Usage
-----
  python run_experiment_3.py [--grids N1 N2 N3 [N1 N2 N3 ...]] [--horizon N]
                             [--v-flutter V] [--no-lmi] [--no-plot]
"""

import time, argparse, csv
import numpy as np
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

from systems.aeroelastic_wing import AeroelasticWing
from tp_transform.tp_model import TPModel
from algorithms.prhpg import PRHPG
from algorithms.lmi_synthesis import synthesise_from_tp
from evaluation.nonlinear_sim import simulate_open_loop, simulate_lmi, _postprocess
from evaluation.metrics import suboptimality_gap
from plotting.comparison_plots import plot_trajectory, format_comparison_table
from config import (Q, R, SIGMA0, N_HORIZON, QUAD_POINTS,
                    TABLE_VERTEX_GRIDS, EXP3_V, EXP3_T_SIM, X0)

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = ROOT / 'figures'
RES_DIR = ROOT / 'results'
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VERTEX_GRIDS = TABLE_VERTEX_GRIDS
DEFAULT_HORIZON      = N_HORIZON
DEFAULT_V_SIM        = 35.0    # m/s — well above flutter; diverges in ~3.3 s
DEFAULT_T_SIM        = 3000    # steps = 6 s — captures divergence + settling


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-parameter PDC simulation (wing has no state-derived scheduling)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_pdc_fixed_p(system, tp_model, K_vertices, x0, T, Q, R,
                           p_fixed, div_tol=1e3, settle_tol=1e-3,
                           settle_window=10, label='P-RHPG'):
    """PDC closed-loop simulation at a fixed operating point.

    Computes K_t = Σ_i α_i(p_fixed) K_i and applies u_t = K_t x_t at
    every step with p held constant at p_fixed.  Uses system.simulate_step
    for state propagation (RK4 integration).

    This is appropriate for the aeroelastic wing where p = (V, δK, δC) is
    an externally commanded airspeed, not a function of state.

    Parameters
    ----------
    system     : AeroelasticWing
    tp_model   : TPModel
    K_vertices : (N_v, m, n)
    x0         : (n,)
    T          : int — steps
    Q, R       : cost matrices
    p_fixed    : (n_p,) — fixed scheduling point
    div_tol    : float  — ‖x‖ threshold for divergence flag

    Returns
    -------
    result : SimResult (via nonlinear_sim._postprocess)
    """
    import time as _time
    t0  = _time.perf_counter()
    n   = len(x0)
    m   = K_vertices.shape[1]
    n_p = len(p_fixed)

    # Compute constant blended gain K̄ = Σ_i α_i(p) K_i
    alpha   = tp_model.weights(p_fixed)
    K_blend = np.einsum('i,imn->mn', alpha, K_vertices)   # (m, n)

    x_traj = np.zeros((T + 1, n))
    u_traj = np.zeros((T, m))
    p_traj = np.tile(np.asarray(p_fixed), (T, 1))
    x_traj[0] = x0
    cost = 0.0

    for t in range(T):
        u_t = K_blend @ x_traj[t]          # (m,)
        u_traj[t] = u_t
        x_traj[t + 1] = system.simulate_step(
            x_traj[t],
            float(u_t[0]) if m == 1 else u_t,
            p_fixed,
        )
        cost += (x_traj[t] @ Q @ x_traj[t]
                 + u_t @ R @ u_t)

        if np.linalg.norm(x_traj[t + 1]) > div_tol * 10:
            x_traj[t + 2:] = div_tol * 10
            break

    cost += x_traj[T] @ Q @ x_traj[T]

    return _postprocess(
        x_traj, u_traj, p_traj,
        Q=Q, R=R, cost_in=cost,
        t0=t0, label=label,
        metadata=dict(N_v=tp_model.n_vertices, Ts=system.Ts,
                      p_fixed=list(p_fixed)),
        div_tol=div_tol,
        settle_tol=settle_tol,
        settle_window=settle_window,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part A: Trajectory at V = EXP3_V (above flutter)
# ─────────────────────────────────────────────────────────────────────────────

def run_trajectory(tp, K_vertices_prhpg, lmi_result, V, T=None, tp_lmi=None, K_lqr=None):
    """Simulate open-loop, P-RHPG, and LMI trajectories at fixed V."""
    print(f'\n{"─"*60}')
    print(f'Trajectory simulations  V={V} m/s, T={T} steps')
    print(f'{"─"*60}')

    T       = T if T is not None else EXP3_T_SIM
    system  = tp.system
    p_fixed = np.array([V, 0.0, 0.0])    # (V, δK=0, δC=0)
    Ts      = system.Ts

    # ── Open-loop (u ≡ 0) ────────────────────────────────────────────────────
    print('  Running open-loop...')
    t0 = time.perf_counter()
    res_open = simulate_open_loop(
        system, X0, T,
        p_fixed=p_fixed,
        div_tol=1e2,
        label='Open-loop',
    )
    print(f'    diverged={res_open.diverged}  '
          f'div_step={res_open.div_step}  '
          f'({time.perf_counter()-t0:.2f} s)')

    # ── P-RHPG (PDC, fixed p) ─────────────────────────────────────────────────
    print('  Running P-RHPG...')
    t0 = time.perf_counter()
    res_prhpg = _simulate_pdc_fixed_p(
        system, tp, K_vertices_prhpg, X0, T,
        Q=Q, R=R, p_fixed=p_fixed,
        div_tol=1e2, settle_tol=1e-3, settle_window=20,
        label='P-RHPG',
    )
    print(f'    stable={res_prhpg.stable}  '
          f'settle_step={res_prhpg.settle_step}  '
          f'cost={res_prhpg.cost:.4e}  '
          f'({time.perf_counter()-t0:.2f} s)')

    # ── LMI-PDC (blended gain at p_fixed) ────────────────────────────────────
    res_lmi = None
    if lmi_result is not None and lmi_result.feasible:
        print('  Running LMI-PDC...')
        # Blend LMI vertex gains at p_fixed: K̄_lmi = Σ_i α_i(p) K_i^lmi
        alpha     = (tp_lmi if tp_lmi is not None else tp).weights(p_fixed)
        K_lmi_bl  = np.einsum('i,imn->mn', alpha, lmi_result.K_vertices)
        t0 = time.perf_counter()
        res_lmi = simulate_lmi(
            system, K_lmi_bl, X0, T,
            Q=Q, R=R, p_fixed=p_fixed,
            div_tol=1e2, settle_tol=1e-3, settle_window=20,
            label='LMI-PDC',
        )
        print(f'    stable={res_lmi.stable}  '
              f'settle_step={res_lmi.settle_step}  '
              f'cost={res_lmi.cost:.4e}  '
              f'({time.perf_counter()-t0:.2f} s)')
    else:
        print('  LMI infeasible — skipping LMI trajectory')

    # ── Nominal LQR (single frozen-point design, unscheduled) ────────────────
    res_lqr = None
    if K_lqr is not None:
        print('  Running nominal LQR...')
        res_lqr = simulate_lmi(
            system, K_lqr, X0, T,
            Q=Q, R=R, p_fixed=p_fixed,
            div_tol=1e2, settle_tol=1e-3, settle_window=20,
            label='LQR (nominal)',
        )
        print(f'    stable={res_lqr.stable}  settle_step={res_lqr.settle_step}  '
              f'cost={res_lqr.cost:.4e}')

    return res_open, res_prhpg, res_lmi, res_lqr


# ─────────────────────────────────────────────────────────────────────────────
# Part B: Comparison table across vertex grids
# ─────────────────────────────────────────────────────────────────────────────

LMI_SOLVER = 'SCS'   # solver used for the submitted results (paper text names SCS)


def _run_one_grid(vertex_grid, horizon, run_lmi=True, run_prhpg=True):
    """Build TP model, run P-RHPG and LMI, return row dicts."""
    N_v_label = int(np.prod(vertex_grid))
    print(f'\n  Grid {vertex_grid}  (N_v={N_v_label})')

    system = AeroelasticWing()
    tp     = TPModel.from_grid(system, vertex_grid)
    solver = PRHPG(tp, Q, R, SIGMA0, quad_points=QUAD_POINTS)
    J_ric, _ = solver.riccati_benchmark()

    # P-RHPG
    if run_prhpg:
        t0_pg = time.perf_counter()
        result = solver.solve(horizon, Q_N='are', method='direct', verbose=False)
        t_pg   = time.perf_counter() - t0_pg
        J_inf_pg, rho_pg = solver.evaluate_infinite_cost(result.K_vertices)
        gap_pg = suboptimality_gap(J_inf_pg, J_ric)
    else:
        class _Dummy: pass
        result = _Dummy(); result.K_vertices = np.zeros((tp.n_vertices, tp.m, tp.n)); result.integrated_cost = np.nan
        t_pg = 0.0; J_inf_pg = rho_pg = gap_pg = np.nan

    print(f'    P-RHPG: J_∞={J_inf_pg:.4e}  gap={gap_pg:.1f}%  '
          f'ρ_max={rho_pg:.4f}  t={t_pg:.2f}s')

    row_pg = dict(
        method     = 'P-RHPG',
        N_v        = tp.n_vertices,
        J_inf      = float(J_inf_pg),
        obj_bound  = float(result.integrated_cost),   # J_N finite-horizon
        J_ric      = float(J_ric),
        gap_pct    = float(gap_pg),
        solve_time = float(t_pg),
        feasible   = True,
        stable     = bool(np.isfinite(J_inf_pg) and rho_pg < 1.0),
    )

    # LMI
    row_lmi = None
    lmi_result = None
    if run_lmi:
        t0_lmi = time.perf_counter()
        lmi_result = synthesise_from_tp(
            tp, Q, R, SIGMA0,
            objective='h2', solver=LMI_SOLVER, verbose=False,
        )
        t_lmi = time.perf_counter() - t0_lmi

        if lmi_result.feasible:
            J_inf_lmi, rho_lmi = solver.evaluate_infinite_cost(
                lmi_result.K_vertices)
            gap_lmi = suboptimality_gap(J_inf_lmi, J_ric)
            # Distinguish truly feasible from numerically inaccurate solutions
            actually_stable = bool(np.isfinite(J_inf_lmi) and rho_lmi < 1.0)
            if actually_stable:
                status_str = f'Tr(W)={lmi_result.objective:.4e}'
            else:
                status_str = f'INACCURATE (solver:{lmi_result.status}, gain unstable ρ={rho_lmi:.4f})'
            print(f'    LMI-PDC: J_∞={J_inf_lmi:.4e}  gap={gap_lmi:.1f}%  '
                  f'ρ_max={rho_lmi:.4f}  {status_str}  t={t_lmi:.2f}s')
        else:
            J_inf_lmi = rho_lmi = gap_lmi = np.nan
            actually_stable = False
            print(f'    LMI-PDC: INFEASIBLE  ({lmi_result.status})  '
                  f't={t_lmi:.2f}s')

        row_lmi = dict(
            method     = 'LMI-PDC',
            N_v        = tp.n_vertices,
            J_inf      = float(J_inf_lmi),
            obj_bound  = float(lmi_result.objective),   # Tr(W) upper bound
            J_ric      = float(J_ric),
            gap_pct    = float(gap_lmi),
            solve_time = float(t_lmi),
            # feasible = SDP claimed success AND gain is actually stabilising
            feasible   = bool(lmi_result.feasible and actually_stable),
            stable     = bool(actually_stable),
        )

    return tp, result, lmi_result, row_pg, row_lmi


def run_comparison_table(vertex_grids, horizon, run_lmi=True, run_prhpg=True):
    """Sweep vertex grids and build comparison table rows."""
    print(f'\n{"─"*60}')
    print(f'Comparison table  N={horizon}  grids={[list(g) for g in vertex_grids]}')
    print(f'{"─"*60}')

    all_rows  = []
    tp_best   = None
    K_best_pg = None
    lmi_best  = None
    tp_lmi_best = None

    for vertex_grid in vertex_grids:
        tp, result, lmi_result, row_pg, row_lmi = _run_one_grid(
            list(vertex_grid), horizon, run_lmi=run_lmi, run_prhpg=run_prhpg)
        all_rows.append(row_pg)
        if row_lmi is not None:
            all_rows.append(row_lmi)

        # Keep last (finest) grid for trajectory plot
        tp_best   = tp
        K_best_pg = result.K_vertices
        # Keep the last FEASIBLE LMI design (with its own TP model for blending)
        if lmi_result is not None and lmi_result.feasible:
            lmi_best, tp_lmi_best = lmi_result, tp

    return all_rows, tp_best, K_best_pg, lmi_best, tp_lmi_best


# ─────────────────────────────────────────────────────────────────────────────
# Save + plot
# ─────────────────────────────────────────────────────────────────────────────

def print_table(all_rows):
    """Print aligned text table to stdout."""
    print(f'\n{"─"*60}')
    print('Comparison table')
    print(f'{"─"*60}')
    hdr = (f"  {'Method':<10} {'Nv':>4} {'J_inf':>13} "
           f"{'Bound':>13} {'J_ric':>13} {'Gap%':>7} "
           f"{'Time':>7} {'Stable':>6}")
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))
    prev_nv = None
    for r in all_rows:
        if prev_nv is not None and r['N_v'] != prev_nv:
            print()
        prev_nv = r['N_v']
        J_inf = r['J_inf']; bound = r['obj_bound']; J_ric = r['J_ric']
        gap   = r['gap_pct']; t = r['solve_time']
        def _f(v): return f'{v:.4e}' if np.isfinite(v) else '    --      '
        print(f"  {r['method']:<10} {r['N_v']:>4} {_f(J_inf):>13} "
              f"{_f(bound):>13} {_f(J_ric):>13} "
              f"{gap:>7.2f} {t:>7.2f} "
              f"{'✓' if r['stable'] else '✗':>6}")


def save_results(all_rows, vertex_grids, horizon, V=35.0, T_sim=3000):
    """Write CSV, LaTeX table, and summary."""
    # CSV
    csv_path = RES_DIR / 'exp3_table.csv'
    fields = ['method', 'N_v', 'J_inf', 'obj_bound', 'J_ric',
              'gap_pct', 'solve_time', 'feasible', 'stable']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f'  Saved CSV  → {csv_path}')

    # LaTeX + verbose text table
    tex_path = RES_DIR / 'exp3_table.tex'
    _, latex_str = format_comparison_table(
        all_rows,
        save_path_csv=None,       # already saved above
        save_path_tex=str(tex_path),
    )

    # Summary
    pg_rows  = [r for r in all_rows if r['method'] == 'P-RHPG']
    lmi_rows = [r for r in all_rows if r['method'] == 'LMI-PDC']

    best_pg  = min((r for r in pg_rows  if np.isfinite(r['J_inf'])),
                   key=lambda r: r['J_inf'], default=None)
    best_lmi = min((r for r in lmi_rows if np.isfinite(r['J_inf'])),
                   key=lambda r: r['J_inf'], default=None)

    lines = [
        '=' * 60,
        'Experiment 3 — Trajectory and Method Comparison',
        '=' * 60,
        f'Horizon N       : {horizon}',
        f'Vertex grids    : {[list(g) for g in vertex_grids]}',
        f'Flutter speed   : {V} m/s',
        f'Sim steps       : {T_sim}  ({T_sim*0.002:.2f} s)',
        '',
        '── P-RHPG ────────────────────────────────────────────────',
    ]
    for r in pg_rows:
        lines.append(
            f'  N_v={r["N_v"]:>3}:  J_∞={r["J_inf"]:.4e}  '
            f'gap={r["gap_pct"]:.1f}%  t={r["solve_time"]:.2f}s'
        )
    if best_pg:
        lines += [f'  Best: N_v={best_pg["N_v"]}, gap={best_pg["gap_pct"]:.2f}%', '']

    lines.append('── LMI-PDC ───────────────────────────────────────────────')
    if lmi_rows:
        for r in lmi_rows:
            feas = '✓' if r['feasible'] else '✗ INFEASIBLE'
            lines.append(
                f'  N_v={r["N_v"]:>3}:  J_∞={r["J_inf"]:.4e}  '
                f'Tr(W)={r["obj_bound"]:.4e}  '
                f'gap={r["gap_pct"]:.1f}%  t={r["solve_time"]:.2f}s  {feas}'
            )
        if best_lmi:
            lines += [f'  Best: N_v={best_lmi["N_v"]}, gap={best_lmi["gap_pct"]:.2f}%']
    else:
        lines.append('  (LMI skipped via --no-lmi)')

    lines.append('=' * 60)
    summary = '\n'.join(lines)
    print('\n' + summary)
    txt_path = RES_DIR / 'exp3_summary.txt'
    txt_path.write_text(summary)
    print(f'\n  Saved TXT  → {txt_path}')


def nominal_lqr_gain(system, p_nom):
    """Single LQR gain at one frozen operating point (unscheduled baseline)."""
    from scipy import linalg
    n = system.n
    S = system.system_matrix(np.asarray(p_nom, float), discrete=True)
    A, B = S[:, :n], S[:, n:]
    P = linalg.solve_discrete_are(A, B, Q, R)
    return -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def lqr_table_row(tp, solver, K_lqr, J_ric, V_nom):
    """Table row for the unscheduled LQR gain: K(p) = K_lqr for all p."""
    Kv = np.tile(K_lqr, (tp.n_vertices, 1, 1))
    J, rho = solver.evaluate_infinite_cost(Kv)
    return dict(method=f'LQR@{V_nom:g}', N_v=1, J_inf=float(J), obj_bound=np.nan,
                J_ric=float(J_ric), gap_pct=float(suboptimality_gap(J, J_ric)) if np.isfinite(J) else np.nan,
                solve_time=0.0, feasible=True, stable=bool(np.isfinite(J) and rho < 1.0))


def make_figure(res_open, res_prhpg, res_lmi, res_lqr=None):
    """Render and save Figure 3."""
    print(f'\n{"─"*60}')
    print('Rendering Figure 3')
    print(f'{"─"*60}')

    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    p_fig3 = FIG_DIR / 'fig3_trajectory.pdf'
    plot_trajectory(
        res_open, res_prhpg,
        state_indices = (0, 2),
        state_labels  = (r'Plunge $h$ [m]', r'Pitch $\alpha$ [rad]'),
        res_lmi       = res_lmi,
        save_path     = p_fig3,
        title         = None,
        zoom_window   = (0.5, 1.5),
        res_extra     = ([(res_lqr, 'LQR (nominal)', '#ff7f0e', '--')]
                         if res_lqr is not None else None),
    )
    return p_fig3


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Experiment 3: trajectory + table')
    p.add_argument('--grids', nargs='+', type=int, default=None,
                   help='Flat list of vertex grid sizes (groups of n_p=3). '
                        'E.g. --grids 2 2 2 3 3 3 for two grids.')
    p.add_argument('--horizon', type=int, default=DEFAULT_HORIZON,
                   help=f'P-RHPG horizon N (default: {DEFAULT_HORIZON})')
    p.add_argument('--v-flutter', type=float, default=DEFAULT_V_SIM,
                   help=f'Airspeed for trajectory (default: {DEFAULT_V_SIM})')
    p.add_argument('--t-sim', type=int, default=DEFAULT_T_SIM,
                   help=f'Simulation steps for trajectory (default: {DEFAULT_T_SIM})')
    p.add_argument('--no-lmi',  action='store_true',
                   help='Skip LMI synthesis (faster)')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip figure rendering')
    p.add_argument('--skip-prhpg', action='store_true',
                   help='LMI-only sweep (P-RHPG rows left NaN, no trajectories)')
    p.add_argument('--lqr', action='store_true',
                   help='Add nominal single-point LQR baseline (row + trajectory)')
    p.add_argument('--v-nom', type=float, default=20.0,
                   help='Design airspeed for the nominal LQR (default 20 m/s)')
    return p.parse_args()


def main():
    args = parse_args()

    # Parse --grids as groups of 3
    if args.grids is not None:
        raw = args.grids
        if len(raw) % 3 != 0:
            raise ValueError('--grids must be groups of 3 (n_p=3 for wing)')
        vertex_grids = [raw[i:i+3] for i in range(0, len(raw), 3)]
    else:
        vertex_grids = [list(g) for g in DEFAULT_VERTEX_GRIDS]

    horizon   = args.horizon
    V         = args.v_flutter
    T_sim     = args.t_sim
    run_lmi   = not args.no_lmi

    print('\nExperiment 3: Trajectory and Method Comparison')
    print(f'  vertex_grids = {vertex_grids}')
    print(f'  horizon N    = {horizon}')
    print(f'  V (flutter)  = {V} m/s')
    print(f'  T_sim        = {T_sim} steps ({T_sim*0.002:.2f} s)')
    print(f'  LMI          = {run_lmi}')

    t_total = time.perf_counter()

    # ── Part B: table sweep (includes building the finest tp + K for Part A) ──
    all_rows, tp_finest, K_prhpg, lmi_finest, tp_lmi = run_comparison_table(
        vertex_grids, horizon, run_lmi=run_lmi, run_prhpg=not args.skip_prhpg)
    if args.skip_prhpg:
        print_table(all_rows)
        csv_path = RES_DIR / 'exp3_lmi_only.csv'
        import csv as _csv
        with open(csv_path, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
        print(f'  Saved LMI-only table -> {csv_path}')
        return dict(all_rows=all_rows)

    K_lqr = None
    if args.lqr:
        K_lqr = nominal_lqr_gain(tp_finest.system, [args.v_nom, 0.0, 0.0])
        solver_f = PRHPG(tp_finest, Q, R, SIGMA0, quad_points=QUAD_POINTS)
        J_ric_f, _ = solver_f.riccati_benchmark()
        all_rows.append(lqr_table_row(tp_finest, solver_f, K_lqr, J_ric_f, args.v_nom))

    print_table(all_rows)

    # ── Part A: trajectory at V above flutter ────────────────────────────────
    res_open, res_prhpg, res_lmi, res_lqr = run_trajectory(
        tp_finest, K_prhpg, lmi_finest, V=V, T=T_sim, tp_lmi=tp_lmi, K_lqr=K_lqr)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('Saving results')
    print(f'{"─"*60}')
    save_results(all_rows, vertex_grids, horizon, V=V, T_sim=T_sim)

    if not args.no_plot:
        make_figure(res_open, res_prhpg, res_lmi, res_lqr)

    print(f'\nExperiment 3 complete in {time.perf_counter()-t_total:.1f} s')

    return dict(
        all_rows   = all_rows,
        tp         = tp_finest,
        K_prhpg    = K_prhpg,
        lmi_result = lmi_finest,
        res_open   = res_open,
        res_prhpg  = res_prhpg,
        res_lmi    = res_lmi,
        res_lqr    = res_lqr,
    )


if __name__ == '__main__':
    main()