"""
Experiment: Complexity Scaling — Direct vs Gradient Descent
============================================================

For each vertex grid, run one full P-RHPG sweep with both methods.
Uses the public API: solver.solve(N, method='direct'/'gd')

Measures:
  - Wall-clock time for full sweep (both methods)
  - Per-stage quadrature time vs solve time (from stage_info)
  - Total GD iterations across all stages
  - Final J_inf for each method

Output:
  - Console table
  - results/exp_complexity.npz
"""

import time
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent
RES_DIR = ROOT / 'results'
RES_DIR.mkdir(exist_ok=True)

from systems.aeroelastic_wing import AeroelasticWing
from tp_transform.tp_model import TPModel
from algorithms.prhpg import PRHPG

# ── Config ────────────────────────────────────────────────────────────────────
GRIDS    = [[2,2,2], [3,2,2], [3,3,2], [3,3,3], [4,3,3]]
N        = 100
Q        = np.diag([10., 1., 100., 1.])
R        = np.array([[1.]])
Sigma0   = np.eye(4)
GD_TOL   = 1e-6
GD_MAXIT = 5000

print("=" * 80)
print("Complexity Scaling: Direct vs Gradient Descent")
print(f"  Horizon N={N}, GD tol={GD_TOL}, GD max_iter={GD_MAXIT}")
print("=" * 80)

sys_obj = AeroelasticWing()
records = []

header = (f"{'Grid':<10} {'Nv':>4} {'Nq':>5} | "
          f"{'Dir t_q':>8} {'Dir t_s':>8} {'Dir tot':>8} {'Dir J∞':>12} | "
          f"{'GD t_q':>8} {'GD t_s':>8} {'GD tot':>8} {'GD iters':>9} {'GD J∞':>12}")
sep = "-" * len(header)
print(header)
print(sep)

for grid in GRIDS:
    tp     = TPModel.from_grid(sys_obj, grid)
    Nv     = tp.n_vertices
    solver = PRHPG(tp, Q, R, Sigma0)
    Nq     = solver.N_q

    row = dict(grid=grid, Nv=Nv, Nq=Nq)

    # ── Direct solve ──────────────────────────────────────────────────────────
    t0     = time.perf_counter()
    res_d  = solver.solve(N, Q_N=None, method='direct', verbose=False)
    t_dir  = time.perf_counter() - t0

    # Extract quadrature vs solve breakdown from stage_info if available
    t_quad_d  = sum(s.get('t_quad', 0.) for s in res_d.stage_info)
    t_solve_d = sum(s.get('t_solve', 0.) for s in res_d.stage_info)

    J_inf_d, rho_d = solver.evaluate_infinite_cost(res_d.K_vertices)
    row.update(dir_tq=t_quad_d, dir_ts=t_solve_d, dir_tot=t_dir, dir_Jinf=J_inf_d)

    # ── Gradient descent ──────────────────────────────────────────────────────
    t0     = time.perf_counter()
    res_g  = solver.solve(N, Q_N=None, method='gd',
                          max_iter=GD_MAXIT, tol=GD_TOL, verbose=False)
    t_gd   = time.perf_counter() - t0

    t_quad_g  = sum(s.get('t_quad', 0.)   for s in res_g.stage_info)
    t_solve_g = sum(s.get('t_solve', 0.)  for s in res_g.stage_info)
    total_iters = sum(s.get('n_iter', 0)  for s in res_g.stage_info)

    J_inf_g, rho_g = solver.evaluate_infinite_cost(res_g.K_vertices)
    row.update(gd_tq=t_quad_g, gd_ts=t_solve_g, gd_tot=t_gd,
               gd_iters=total_iters, gd_Jinf=J_inf_g)

    records.append(row)

    print(f"{str(grid):<10} {Nv:>4} {Nq:>5} | "
          f"{t_quad_d:>8.2f} {t_solve_d:>8.2f} {t_dir:>8.2f} {J_inf_d:>12.4e} | "
          f"{t_quad_g:>8.2f} {t_solve_g:>8.2f} {t_gd:>8.2f} {total_iters:>9} {J_inf_g:>12.4e}")

print(sep)
print("All times in seconds. t_q = quadrature build, t_s = stage solve.")
print("Note: if t_q/t_s are 0, stage_info does not include timing keys.")

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(RES_DIR / 'exp_complexity.npz',
         Nvs      = np.array([r['Nv']        for r in records]),
         Nqs      = np.array([r['Nq']        for r in records]),
         dir_tq   = np.array([r['dir_tq']    for r in records]),
         dir_ts   = np.array([r['dir_ts']    for r in records]),
         dir_tot  = np.array([r['dir_tot']   for r in records]),
         dir_Jinf = np.array([r['dir_Jinf']  for r in records]),
         gd_tq    = np.array([r['gd_tq']     for r in records]),
         gd_ts    = np.array([r['gd_ts']     for r in records]),
         gd_tot   = np.array([r['gd_tot']    for r in records]),
         gd_iters = np.array([r['gd_iters']  for r in records]),
         gd_Jinf  = np.array([r['gd_Jinf']   for r in records]),
         N=N, GD_TOL=GD_TOL)

print(f"\nSaved -> {RES_DIR}/exp_complexity.npz")