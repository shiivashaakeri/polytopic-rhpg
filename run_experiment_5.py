"""
Experiment 5 (camera-ready additions)
=====================================
(A) Nominal-LQR baseline (Rev. 14): a single LQR gain designed at one frozen
    operating point, deployed unscheduled over all of Omega.  Reports J_inf,
    rho_max, stable fraction of Omega, and a sweep over the design airspeed.
    Also simulates it at V = 35 m/s for Fig. 3.
(B) A-posteriori uniform quadratic Lyapunov certificate for the deployed
    P-RHPG gain (Rev. 3 / new Proposition):  find W > 0, rho < 1 with
        Acl(p)' W Acl(p) <= rho^2 W   for all p on a fine grid,
    then the constants a = max ||Acl(p)||, kappa_W, L_P (finite-difference
    Lipschitz estimate of P_inf(p)) and C = a^2 L_P kappa_W / (1 - rho^2).
(C) Bound sanity check on an airspeed ramp: |J_traj - x0' P_inf(p0) x0| vs C*delta*||x0||^2.

Usage:  python run_experiment_5.py [--grid 3 2 2] [--horizon 100]
"""
import argparse, time, json
import numpy as np
from pathlib import Path
from scipy import linalg
import cvxpy as cp

from systems.aeroelastic_wing import AeroelasticWing
from tp_transform.tp_model import TPModel
from tp_transform.sampling import ParameterGrid
from algorithms.prhpg import PRHPG
from evaluation.nonlinear_sim import simulate_open_loop, simulate_lmi
from evaluation.metrics import suboptimality_gap
from run_experiment_3 import _simulate_pdc_fixed_p
from config import Q, R, SIGMA0, QUAD_POINTS, X0, V_NOM

ROOT = Path(__file__).parent
RES = ROOT / 'results'; RES.mkdir(exist_ok=True)


def lyap_P(Acl, S, tol=1e-12, max_iter=5000):
    P = S.copy()
    for _ in range(max_iter):
        Pn = S + Acl.T @ P @ Acl
        if np.max(np.abs(Pn - P)) < tol:
            return Pn
        P = Pn
    return P


def blended_Acl_grid(tp, K_vertices, grid):
    """Acl(p) = A(p) + B(p) K(p) on grid points, using the TRUE system A,B."""
    sysm = tp.system
    n = tp.n
    Acl = []
    Kb = []
    for p in grid.grid_points:
        S = sysm.system_matrix(p, discrete=True)
        A, B = S[:, :n], S[:, n:]
        K = np.einsum('i,imn->mn', tp.weights(p), K_vertices)
        Acl.append(A + B @ K); Kb.append(K)
    return np.array(Acl), np.array(Kb)


# ---------------------------------------------------------------- Part A
def lqr_gain(system, p):
    n = system.n
    S = system.system_matrix(np.asarray(p, float), discrete=True)
    A, B = S[:, :n], S[:, n:]
    P = linalg.solve_discrete_are(A, B, Q, R)
    return -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def eval_constant_gain(solver, K):
    Kv = np.tile(K, (solver.Nv, 1, 1))           # K(p) = K for all p
    J, rho = solver.evaluate_infinite_cost(Kv)
    # stable fraction over quadrature grid
    n_stab = 0
    for q in range(solver.N_q):
        Acl = solver.A_quad[q] + solver.B_quad[q] @ K
        n_stab += np.max(np.abs(np.linalg.eigvals(Acl))) < 1.0
    return J, rho, n_stab / solver.N_q, Kv


def part_A(tp, solver, J_ric, V_test, T_sim):
    out = {}
    print('\n== Part A: nominal LQR baseline ==')
    for Vd in [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]:
        K = lqr_gain(tp.system, [Vd, 0.0, 0.0])
        J, rho, frac, _ = eval_constant_gain(solver, K)
        gap = suboptimality_gap(J, J_ric) if np.isfinite(J) else np.nan
        # largest V (dK=dC=0) still stable
        Vs = np.linspace(5, 35, 301); Vmax_stab = None
        for V in Vs:
            S = tp.system.system_matrix(np.array([V, 0, 0]), discrete=True)
            A, B = S[:, :4], S[:, 4:]
            if np.max(np.abs(np.linalg.eigvals(A + B @ K))) >= 1: break
            Vmax_stab = V
        out[Vd] = dict(J_inf=float(J), rho_max=float(rho), stable_frac=float(frac),
                       gap_pct=float(gap), Vmax_stable_nominal=Vmax_stab)
        print(f'  LQR @ V={Vd:4.1f}: J_inf={J:.4e} gap={gap:6.1f}%  rho_max={rho:.3f}  '
              f'stable on {100*frac:5.1f}% of quad grid; stable up to V={Vmax_stab}')
    # trajectory at V_test for the nominal design
    K_nom = lqr_gain(tp.system, [V_NOM, 0.0, 0.0])
    p_fixed = np.array([V_test, 0.0, 0.0])
    res = simulate_lmi(tp.system, K_nom, X0, T_sim, Q=Q, R=R, p_fixed=p_fixed,
                       div_tol=1e2, settle_tol=1e-3, settle_window=20, label='LQR (nominal)')
    print(f'  Trajectory at V={V_test}: stable={res.stable} diverged={res.diverged} '
          f'div_step={res.div_step} cost={res.cost:.4e}')
    return out, K_nom, res


# ---------------------------------------------------------------- Part B
def certificate_polytopic(Acl, alpha, rho2, wmax=1e4, solver='CLARABEL'):
    """Find W_i > 0 with W(p)=sum_i alpha_i(p) W_i and Acl(p)'W(p)Acl(p) <= rho2 W(p) on the grid."""
    Nq, n, _ = Acl.shape; Nv = alpha.shape[1]
    Ws = [cp.Variable((n, n), PSD=True) for _ in range(Nv)]
    cons = [W >> np.eye(n) for W in Ws] + [W << wmax * np.eye(n) for W in Ws]
    for q in range(Nq):
        Wq = sum(alpha[q, i] * Ws[i] for i in range(Nv))
        cons.append(rho2 * Wq - Acl[q].T @ Wq @ Acl[q] >> 0)
    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver=solver)
    except Exception:
        return None
    if prob.status in ('optimal', 'optimal_inaccurate'):
        return np.array([W.value for W in Ws])
    return None


def lipschitz_along_V(tp, fn, nV=301):
    """max ||fn(p_{k+1}) - fn(p_k)||_2 / dV along the line dK=dC=0."""
    Vs = np.linspace(tp.system.param_bounds[0, 0], tp.system.param_bounds[0, 1], nV)
    vals = [fn(np.array([V, 0.0, 0.0])) for V in Vs]
    dV = Vs[1] - Vs[0]
    return max(np.linalg.norm(vals[k+1] - vals[k], 2) / dV for k in range(nV - 1))


def part_B(tp, K_vertices, eval_pts, tag, common=False):
    print(f'\n== Part B: Lyapunov certificate for deployed gain ({tag}) ==')
    sysm = tp.system; n = tp.n
    grid = ParameterGrid(sysm.param_bounds, n_points=eval_pts)
    Acl, Kb = blended_Acl_grid(tp, K_vertices, grid)
    alpha = np.array([tp.weights(p) for p in grid.grid_points])
    if common:
        alpha = np.ones((alpha.shape[0], 1))
    rho_max = max(np.max(np.abs(np.linalg.eigvals(A))) for A in Acl)
    a = max(np.linalg.norm(A, 2) for A in Acl)
    print(f'  grid {eval_pts} ({grid.total_points} pts): rho_max={rho_max:.5f}, a=max||Acl||_2={a:.4f}')

    lo, hi = rho_max**2, 1.0 - 1e-7
    best = None
    t0 = time.perf_counter()
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        Ws = certificate_polytopic(Acl, alpha, mid)
        if Ws is not None: hi, best = mid, (mid, Ws)
        else: lo = mid
    print(f'  bisection done in {time.perf_counter()-t0:.1f}s')
    if best is None:
        print('  NO certificate found'); return None
    rho2, Ws = best
    Wfun = lambda p: np.einsum('i,imn->mn', (np.ones(1) if common else tp.weights(p)), Ws)
    Wg = np.einsum('qi,imn->qmn', alpha, Ws)
    evW = np.array([np.linalg.eigvalsh(W) for W in Wg])
    w_min, w_max = evW[:, 0].min(), evW[:, -1].max(); kappa_W = w_max / w_min
    viol = max(np.max(np.linalg.eigvalsh(Acl[q].T @ Wg[q] @ Acl[q] - rho2 * Wg[q])) for q in range(len(Acl)))

    def Pfun(p):
        S = sysm.system_matrix(p, discrete=True); A, B = S[:, :n], S[:, n:]
        K = np.einsum('i,imn->mn', tp.weights(p), K_vertices)
        return lyap_P(A + B @ K, Q + K.T @ R @ K)
    L_P = lipschitz_along_V(tp, Pfun)
    L_W = lipschitz_along_V(tp, Wfun)
    Pg = np.array([lyap_P(A, Q + k.T @ R @ k) for A, k in zip(Acl, Kb)])
    evP = np.array([np.linalg.eigvalsh(P) for P in Pg])
    print(f'  certified rho={np.sqrt(rho2):.5f}  kappa_W={kappa_W:.3e}  max viol={viol:.2e}')
    print(f'  L_P(V)={L_P:.3e}  L_W(V)={L_W:.3e}  P_min={evP[:,0].min():.3e}  P_max={evP[:,-1].max():.3e}')
    out = dict(rho_max=float(rho_max), a=float(a), rho_cert=float(np.sqrt(rho2)), kappa_W=float(kappa_W),
               w_min=float(w_min), L_P=float(L_P), L_W=float(L_W), max_violation=float(viol),
               grid=list(eval_pts), common=common, rows=[])
    # bound constant for a range of parameter rates delta [m/s per step]
    print('  delta [m/s/step]  rho_tilde     C=L_P*kappa*rt2/(1-rt2)   rel. bound at ||x0||^2 P_min')
    for delta in [1e-5, 1e-4, 1e-3, 1e-2]:
        rt2 = rho2 + a**2 * L_W * delta / w_min
        if rt2 >= 1:
            print(f'  {delta:9.1e}   rho_tilde>=1 (bound void)'); out['rows'].append(dict(delta=delta, void=True)); continue
        C = L_P * kappa_W * rt2 / (1 - rt2)
        print(f'  {delta:9.1e}   {np.sqrt(rt2):.6f}   {C:.3e}   {C*delta/evP[:,0].min():.3e}')
        out['rows'].append(dict(delta=delta, rho_tilde=float(np.sqrt(rt2)), C=float(C)))
    out['Ws'] = Ws.tolist()
    return out


# ---------------------------------------------------------------- Part C
def part_C(tp, K_vertices, cert, V0, V1, ramp_times):
    print('\n== Part C: bound check on airspeed ramps ==')
    sysm = tp.system; n = tp.n; Ts = sysm.Ts
    rows = []
    for T_ramp in ramp_times:
        T = int(max(6.0, 2 * T_ramp) / Ts)
        x = X0.copy(); J = 0.0
        for t in range(T):
            V = V0 + (V1 - V0) * min(1.0, t * Ts / T_ramp)
            p = np.array([V, 0.0, 0.0])
            K = np.einsum('i,imn->mn', tp.weights(p), K_vertices)
            S = sysm.system_matrix(p, discrete=True); A, B = S[:, :n], S[:, n:]
            u = K @ x
            J += x @ Q @ x + u @ R @ u
            x = A @ x + B @ u
        # frozen cost at p0
        p0 = np.array([V0, 0, 0])
        S = sysm.system_matrix(p0, discrete=True); A, B = S[:, :n], S[:, n:]
        K0 = np.einsum('i,imn->mn', tp.weights(p0), K_vertices)
        P0 = lyap_P(A + B @ K0, Q + K0.T @ R @ K0)
        J_frozen = X0 @ P0 @ X0
        delta = (V1 - V0) / T_ramp * Ts
        rt2 = cert['rho_cert']**2 + cert['a']**2 * cert['L_W'] * delta / cert['w_min']
        bound = (cert['L_P'] * cert['kappa_W'] * rt2 / (1 - rt2)) * delta * X0 @ X0 if rt2 < 1 else np.inf
        rows.append(dict(T_ramp=T_ramp, delta=delta, J_traj=float(J), J_frozen=float(J_frozen),
                         gap=float(abs(J - J_frozen)), rel_gap=float(abs(J - J_frozen)/J_frozen), bound=float(bound)))
        print(f'  ramp {V0}->{V1} in {T_ramp:5.2f}s: delta={delta:.2e}  J_traj={J:.5e}  '
              f'J_frozen(p0)={J_frozen:.5e}  rel gap={abs(J-J_frozen)/J_frozen:.3e}  bound={bound:.3e}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', nargs=3, type=int, default=[3, 2, 2])
    ap.add_argument('--horizon', type=int, default=1000)
    ap.add_argument('--v-test', type=float, default=35.0)
    ap.add_argument('--t-sim', type=int, default=3000)
    ap.add_argument('--eval-pts', nargs=3, type=int, default=[20, 8, 8])
    ap.add_argument('--skip-c', action='store_true')
    ap.add_argument('--k-file', type=str, default=None, help='load deployed vertex gains (.npy) instead of solving')
    ap.add_argument('--solve-only', action='store_true')
    args = ap.parse_args()

    t_all = time.perf_counter()
    system = AeroelasticWing()
    tp = TPModel.from_grid(system, args.grid)
    solver = PRHPG(tp, Q, R, SIGMA0, quad_points=QUAD_POINTS)
    J_ric, _ = solver.riccati_benchmark()
    print(f'grid {args.grid} Nv={tp.n_vertices}, N={args.horizon}, J_ric={J_ric:.4e}')

    t0 = time.perf_counter()
    if args.k_file:
        K = np.load(args.k_file)
    else:
        res = solver.solve(args.horizon, Q_N='are', method='direct', verbose=False)
        K = res.K_vertices
        np.save(RES / f'exp5_K_{"".join(map(str,args.grid))}_N{args.horizon}.npy', K)
    if args.solve_only:
        print('saved gains; exiting'); return
    J_pg, rho_pg = solver.evaluate_infinite_cost(K)
    print(f'P-RHPG: J_inf={J_pg:.4e} gap={suboptimality_gap(J_pg, J_ric):.2f}% rho_max(quad)={rho_pg:.4f} '
          f'({time.perf_counter()-t0:.1f}s)')

    A_out, K_nom, res_lqr = part_A(tp, solver, J_ric, args.v_test, args.t_sim)
    cert = part_B(tp, K, args.eval_pts, tag=f'grid {args.grid}, N={args.horizon}')
    C_out = None
    if cert is not None and not args.skip_c:
        C_out = part_C(tp, K, cert, V0=20.0, V1=35.0, ramp_times=[0.3, 1.0, 3.0, 10.0])

    # P-RHPG and open-loop trajectories at V_test for the figure
    p_fixed = np.array([args.v_test, 0, 0])
    res_open = simulate_open_loop(system, X0, args.t_sim, p_fixed=p_fixed, div_tol=1e2, label='Open-loop')
    res_pg = _simulate_pdc_fixed_p(system, tp, K, X0, args.t_sim, Q, R, p_fixed,
                                   div_tol=1e2, settle_tol=1e-3, settle_window=20)
    np.savez(RES / 'exp5_trajectories.npz',
             x_open=res_open.x_traj, x_prhpg=res_pg.x_traj, x_lqr=res_lqr.x_traj,
             u_prhpg=res_pg.u_traj, u_lqr=res_lqr.u_traj, Ts=system.Ts,
             div_step_open=res_open.div_step if res_open.div_step is not None else -1,
             div_step_lqr=res_lqr.div_step if res_lqr.div_step is not None else -1,
             K_prhpg=K, K_lqr_nom=K_nom)
    summary = dict(grid=args.grid, horizon=args.horizon, J_ric=float(J_ric),
                   prhpg=dict(J_inf=float(J_pg), rho_max=float(rho_pg)),
                   lqr_sweep={str(k): v for k, v in A_out.items()},
                   certificate=cert, ramp_check=C_out)
    (RES / 'exp5_summary.json').write_text(json.dumps(summary, indent=1))
    print(f'\nsaved results/exp5_summary.json and exp5_trajectories.npz  '
          f'(total {time.perf_counter()-t_all:.1f}s)')


if __name__ == '__main__':
    main()
