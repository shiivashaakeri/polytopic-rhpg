"""
Evaluation Metrics
==================

Pure-numpy utility functions used by all evaluation modules and
experiment runners.  No imports from other empty files — this is a
leaf module with zero circular-dependency risk.

Functions
---------
spectral_radii_grid        Per-point ρ(Acl(p_q)) over a parameter grid
vertex_spectral_radii      ρ(A_i) for each open-loop vertex
closed_loop_spectral_radii ρ(Acl(p; K)) at each quadrature point
suboptimality_gap          (J_inf - J_ric) / J_ric  [%]
stability_margin           1 - rho_max  (> 0 iff stable)
trajectory_cost            Σ x'Qx + u'Ru along a recorded trajectory
is_stable                  rho_max < 1 check with optional tolerance
convergence_rate           Geometric fit to J_N* vs N data
summarise_result           Flat dict of key scalars from a PRHPGResult
elapsed                    Context-manager for wall-clock timing
"""

import time
import numpy as np
from contextlib import contextmanager


# ── Spectral radius utilities ─────────────────────────────────────────────────


def spectral_radii_grid(A_quad, B_quad, alpha_quad, K_vertices):
    """Spectral radius of the closed-loop matrix at every quadrature point.

    Computes
        Acl(p_q) = A(p_q) + B(p_q) K(p_q),
        K(p_q)   = Σ_i α_i(p_q) K_i
    for each quadrature point q and returns ρ(Acl(p_q)).

    Parameters
    ----------
    A_quad      : (N_q, n, n)    — system A matrices at quadrature points
    B_quad      : (N_q, n, m)    — system B matrices at quadrature points
    alpha_quad  : (N_q, N_v)     — weighting functions at quadrature points
    K_vertices  : (N_v, m, n)    — vertex gain matrices

    Returns
    -------
    radii : (N_q,) array — ρ(Acl(p_q)) for each q
    """
    N_q = A_quad.shape[0]
    radii = np.empty(N_q)
    for q in range(N_q):
        K_blend = np.einsum("i,imn->mn", alpha_quad[q], K_vertices)
        Acl = A_quad[q] + B_quad[q] @ K_blend
        radii[q] = np.max(np.abs(np.linalg.eigvals(Acl)))
    return radii


def vertex_spectral_radii(A_vertices):
    """Open-loop spectral radius of each vertex matrix A_i.

    Parameters
    ----------
    A_vertices : (N_v, n, n) array  or  list of (n, n) arrays

    Returns
    -------
    radii : (N_v,) array — ρ(A_i)
    """
    A_vertices = np.asarray(A_vertices)
    N_v = A_vertices.shape[0]
    radii = np.empty(N_v)
    for i in range(N_v):
        radii[i] = np.max(np.abs(np.linalg.eigvals(A_vertices[i])))
    return radii


def closed_loop_spectral_radii(A_quad, B_quad, alpha_quad, K_vertices):
    """Alias for spectral_radii_grid — closed-loop radii at quadrature pts.

    Identical computation; named separately for readability in experiment
    scripts that contrast open-loop vertex radii with closed-loop radii.
    """
    return spectral_radii_grid(A_quad, B_quad, alpha_quad, K_vertices)


# ── Cost metrics ──────────────────────────────────────────────────────────────


def suboptimality_gap(J_inf, J_ric):
    """Relative suboptimality gap (%).

    gap = (J_inf - J_ric) / J_ric * 100

    Returns NaN if either input is NaN or J_ric <= 0.

    Parameters
    ----------
    J_inf : float — infinite-horizon cost of the PDC controller
    J_ric : float — pointwise Riccati lower bound

    Returns
    -------
    gap : float [%]
    """
    if np.isnan(J_inf) or np.isnan(J_ric) or J_ric <= 0.0:
        return np.nan
    return (J_inf - J_ric) / J_ric * 100.0


def trajectory_cost(x_traj, u_traj, Q, R, Q_N=None):
    """Compute LQR cost along a recorded state-input trajectory.

        J = Σ_{t=0}^{T-1} (x_t' Q x_t + u_t' R u_t) + x_T' Q_N x_T

    Parameters
    ----------
    x_traj : (T+1, n) array — state trajectory
    u_traj : (T, m) array   — input trajectory
    Q      : (n, n) array   — stage state cost
    R      : (m, m) array   — stage input cost
    Q_N    : (n, n) array or None — terminal cost (default: Q)

    Returns
    -------
    cost : float
    """
    if Q_N is None:
        Q_N = Q
    T = u_traj.shape[0]
    cost = 0.0
    for t in range(T):
        cost += x_traj[t] @ Q @ x_traj[t] + u_traj[t] @ R @ u_traj[t]
    cost += x_traj[T] @ Q_N @ x_traj[T]
    return float(cost)


# ── Stability metrics ─────────────────────────────────────────────────────────


def stability_margin(radii):
    """Stability margin: 1 - max ρ(Acl).

    Positive ⟹ all closed-loop poles strictly inside the unit disk.

    Parameters
    ----------
    radii : (N_q,) array or float — spectral radii

    Returns
    -------
    margin : float  (> 0 iff stable)
    """
    return float(1.0 - np.max(radii))


def is_stable(radii, tol=0.0):
    """Check whether all spectral radii are strictly below 1.

    Parameters
    ----------
    radii : array-like
    tol   : float — allow ρ ≤ 1 - tol (default: strict, tol=0)

    Returns
    -------
    stable : bool
    """
    return bool(np.max(radii) < 1.0 - tol)


# ── Convergence analysis ──────────────────────────────────────────────────────


def convergence_rate(Ns, J_infs):
    """Fit a geometric decay to J_N* vs N data.

    Models J_N* ≈ J_bar + C * r^N and estimates the decay rate r via
    linear regression on log(J_N* - J_bar) ≈ log(C) + N*log(r),
    where J_bar is taken as the minimum observed cost.

    Parameters
    ----------
    Ns     : (K,) array-like — horizon values
    J_infs : (K,) array-like — corresponding infinite-horizon costs

    Returns
    -------
    rate : float — estimated decay rate r  (< 1 means geometric convergence)
    C    : float — pre-factor
    J_bar: float — estimated limit (min observed J_inf)
    """
    Ns = np.asarray(Ns, dtype=float)
    J_infs = np.asarray(J_infs, dtype=float)

    valid = np.isfinite(J_infs)
    Ns, J_infs = Ns[valid], J_infs[valid]

    if len(Ns) < 3:
        return np.nan, np.nan, float(np.nanmin(J_infs))

    J_bar = np.min(J_infs)
    delta = J_infs - J_bar

    # Keep only points with positive delta for log
    pos = delta > 1e-12 * max(abs(J_bar), 1.0)
    if np.sum(pos) < 2:
        return np.nan, np.nan, J_bar

    log_delta = np.log(delta[pos])
    N_pos = Ns[pos]

    # Linear regression: log_delta = log(C) + N * log(r)
    A = np.column_stack([np.ones_like(N_pos), N_pos])
    coeffs, *_ = np.linalg.lstsq(A, log_delta, rcond=None)
    log_C, log_r = coeffs
    return float(np.exp(log_r)), float(np.exp(log_C)), float(J_bar)


def detect_convergence(J_infs, rtol=1e-4, window=3):
    """Find the first horizon index at which J_N* is considered converged.

    Convergence declared when the max relative change over the last
    `window` steps is below `rtol`.

    Parameters
    ----------
    J_infs : (K,) array-like — cost values indexed by horizon
    rtol   : float           — relative tolerance
    window : int             — number of consecutive steps to check

    Returns
    -------
    idx : int or None — index into J_infs (None if not converged)
    """
    J = np.asarray(J_infs, dtype=float)
    for k in range(window, len(J)):
        chunk = J[k - window : k + 1]
        if np.any(~np.isfinite(chunk)):
            continue
        rel_change = np.max(np.abs(np.diff(chunk))) / max(abs(chunk[-1]), 1e-15)
        if rel_change < rtol:
            return k
    return None


# ── Result summarisation ──────────────────────────────────────────────────────


def summarise_result(result, solver, label=""):
    """Extract a flat dict of key scalars from a PRHPGResult + PRHPG solver.

    Parameters
    ----------
    result : PRHPGResult
    solver : PRHPG
    label  : str — optional identifier (e.g. 'N=100' or 'Nv=12')

    Returns
    -------
    d : dict with keys:
        label, J_N, J_inf, J_ric, gap_pct, rho_max, stable,
        margin, n_vertices, n_quad, n_horizon
    """
    J_N = float(result.integrated_cost)
    J_inf, rho_max = solver.evaluate_infinite_cost(result.K_vertices)
    J_ric, _ = solver.riccati_benchmark()
    gap = suboptimality_gap(J_inf, J_ric)
    margin = stability_margin(np.array([rho_max]))
    radii_all = spectral_radii_grid(solver.A_quad, solver.B_quad, solver.alpha_quad, result.K_vertices)

    return dict(
        label=label,
        J_N=J_N,
        J_inf=float(J_inf),
        J_ric=float(J_ric),
        gap_pct=float(gap),
        rho_max=float(rho_max),
        rho_mean=float(np.mean(radii_all)),
        stable=bool(is_stable(radii_all)),
        margin=float(margin),
        n_vertices=solver.Nv,
        n_quad=solver.N_q,
        n_horizon=len(result.cost_history),
    )


# ── Timing ────────────────────────────────────────────────────────────────────


@contextmanager
def elapsed(label=""):
    """Context manager that prints (and yields) wall-clock time.

    Usage
    -----
        with elapsed('P-RHPG N=100') as t:
            result = solver.solve(100)
        print(f'Took {t.seconds:.2f} s')
    """

    class _Timer:
        seconds = None

    timer = _Timer()
    t0 = time.perf_counter()
    try:
        yield timer
    finally:
        timer.seconds = time.perf_counter() - t0
        if label:
            print(f"  [{label}] {timer.seconds:.3f} s")


def timed(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and return (result, elapsed_seconds).

    Parameters
    ----------
    fn : callable
    *args, **kwargs : forwarded to fn

    Returns
    -------
    result  : return value of fn
    elapsed : float — wall-clock seconds
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("metrics.py — Self-test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n, m, N_v, N_q = 4, 1, 8, 72

    # Random stable closed-loop (scale A to have rho < 1)
    A_quad = np.stack([0.9 * rng.standard_normal((n, n)) / n for _ in range(N_q)])
    B_quad = np.stack([rng.standard_normal((n, m)) for _ in range(N_q)])
    alpha_quad = rng.dirichlet(np.ones(N_v), size=N_q)  # (N_q, N_v), rows sum to 1
    K_vertices = rng.standard_normal((N_v, m, n)) * 0.01
    A_vertices = rng.standard_normal((N_v, n, n)) * 0.5

    # --- spectral_radii_grid ---
    radii = spectral_radii_grid(A_quad, B_quad, alpha_quad, K_vertices)
    assert radii.shape == (N_q,), f"Expected ({N_q},), got {radii.shape}"
    assert np.all(radii >= 0), "Spectral radii must be non-negative"
    print(f"\nspectral_radii_grid: shape={radii.shape}, max={radii.max():.4f}, min={radii.min():.4f}  ✓")

    # --- vertex_spectral_radii ---
    v_radii = vertex_spectral_radii(A_vertices)
    assert v_radii.shape == (N_v,)
    print(f"vertex_spectral_radii: shape={v_radii.shape}, max={v_radii.max():.4f}  ✓")

    # --- stability_margin / is_stable ---
    margin = stability_margin(radii)
    stable = is_stable(radii)
    print(f"stability_margin: {margin:.4f},  is_stable: {stable}  ✓")

    # --- suboptimality_gap ---
    gap = suboptimality_gap(11.0, 10.0)
    assert abs(gap - 10.0) < 1e-12, f"Expected 10.0%, got {gap}"
    assert np.isnan(suboptimality_gap(np.nan, 10.0))
    assert np.isnan(suboptimality_gap(10.0, 0.0))
    print(f"suboptimality_gap(11, 10) = {gap:.2f}%  ✓")

    # --- trajectory_cost ---
    T_len = 100
    x_traj = rng.standard_normal((T_len + 1, n))
    u_traj = rng.standard_normal((T_len, m))
    Q_mat = np.eye(n)
    R_mat = np.eye(m)
    cost = trajectory_cost(x_traj, u_traj, Q_mat, R_mat)
    # Manual check
    manual = (
        sum(x_traj[t] @ Q_mat @ x_traj[t] + u_traj[t] @ R_mat @ u_traj[t] for t in range(T_len))
        + x_traj[T_len] @ Q_mat @ x_traj[T_len]
    )
    assert abs(cost - manual) < 1e-10 * abs(manual), f"cost mismatch: {cost} vs {manual}"
    print(f"trajectory_cost: {cost:.4e}  ✓")

    # --- convergence_rate ---
    Ns = np.array([10, 20, 30, 50, 75, 100, 150, 200], dtype=float)
    r_true, C_true, J_bar_true = 0.92, 5.0, 100.0
    J_infs = J_bar_true + C_true * r_true**Ns + rng.normal(0, 1e-6, len(Ns))
    r_fit, C_fit, J_bar_fit = convergence_rate(Ns, J_infs)
    assert abs(r_fit - r_true) < 0.02, f"Rate fit off: {r_fit:.4f} vs {r_true}"
    print(f"convergence_rate: r_fit={r_fit:.4f} (true={r_true})  ✓")

    # --- detect_convergence ---
    J_conv = np.array([120.0, 110.0, 105.0, 102.0, 101.0, 100.5, 100.1, 100.05, 100.02, 100.01])
    idx = detect_convergence(J_conv, rtol=1e-3, window=3)
    assert idx is not None, "Should detect convergence"
    print(f"detect_convergence: idx={idx}  ✓")

    # --- timing ---
    with elapsed("sleep test") as t:
        time.sleep(0.05)
    assert 0.04 < t.seconds < 0.2, f"Unexpected elapsed: {t.seconds}"
    print(f"elapsed: {t.seconds:.3f} s  ✓")

    result_val, dt = timed(lambda: sum(range(10000)))
    assert result_val == sum(range(10000))
    print(f"timed: result={result_val}, dt={dt:.4f} s  ✓")

    print("\nAll tests passed ✓")
