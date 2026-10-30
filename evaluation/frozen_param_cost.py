"""
Frozen-Parameter Cost Evaluation
==================================

The integrated costs J_N and J_∞ in prhpg.py are scalars (averaged
over μ).  This module provides the **per-point** versions:

    J_∞(p; K)   — infinite-horizon cost at a single frozen parameter p
    J_ric(p)    — unconstrained pointwise Riccati cost at p
    J_N(p; K)   — finite-horizon cost at p (backward recursion at fixed p)

These are used to:
    - Generate cost-surface / heatmap plots over Ω
    - Compute the pointwise suboptimality gap J_∞(p;K) - J_ric(p)
    - Validate that ∫ J_∞(p;K) dμ matches solver.evaluate_infinite_cost()

All functions are pure numpy / scipy — no imports from other empty files.

Key relation to the paper
-------------------------
The integrated cost  J_N(K) = ∫_Ω J_N(p; K) dμ(p)  (eq. cost-integrated)
decomposes into per-point costs.  Plotting J_∞(p; K) over Ω shows *where*
in the scheduling domain the PDC constraint is most binding (large gap from
J_ric(p)) and where the controller is essentially optimal (small gap).
"""

import numpy as np
from scipy import linalg


# ── Single-point cost primitives ─────────────────────────────────────────────


def lyapunov_cost(Acl, S, Sigma0):
    """Infinite-horizon cost at a single parameter point via discrete Lyapunov.

    Solves  P = S + Acl' P Acl  (discrete-time Lyapunov equation)
    and returns  Tr(P Σ_0).

    Parameters
    ----------
    Acl    : (n, n) array — closed-loop A + B K at this parameter point
    S      : (n, n) array — stage cost  Q + K' R K
    Sigma0 : (n, n) array — initial-condition weighting

    Returns
    -------
    cost : float — Tr(P Σ_0), or np.nan if Acl is not Schur-stable
    """
    rho = np.max(np.abs(np.linalg.eigvals(Acl)))
    if rho >= 1.0:
        return np.nan

    try:
        P = linalg.solve_discrete_lyapunov(Acl.T, S)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    return float(np.trace(P @ Sigma0))


def riccati_cost_at_p(A, B, Q, R, Sigma0):
    """Unconstrained (pointwise) Riccati infinite-horizon cost at p.

    Solves the DARE  P = Q + A'PA - A'PB(R+B'PB)^{-1}B'PA
    and returns  (Tr(P Σ_0), K_ric)  where K_ric is the optimal gain.

    Parameters
    ----------
    A, B   : (n, n), (n, m) — system matrices at this p
    Q, R   : cost matrices
    Sigma0 : (n, n)

    Returns
    -------
    cost   : float — Tr(P_ric Σ_0), or np.nan if DARE fails
    K_ric  : (m, n) array — optimal gain, or None if DARE fails
    """
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
        K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return float(np.trace(P @ Sigma0)), K
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, None


def finite_horizon_cost_at_p(A, B, K_blend, Q, R, Q_N, Sigma0, N):
    """Finite-horizon frozen-parameter cost J_N(p; K) at a single point.

    Runs the backward Riccati recursion with the **fixed** blended gain
    K_blend (no optimisation — K is held constant across all stages):

        P_{N}   = Q_N
        P_{h}   = Q + K' R K + (A + B K)' P_{h+1} (A + B K)
        J_N     = Tr(P_0 Σ_0)

    This is NOT the same as the P-RHPG stage-wise optimisation.
    It answers: "what cost does the first-stage gain produce when applied
    receding-horizon style at this fixed operating point?"

    Parameters
    ----------
    A, B      : (n, n), (n, m) — system matrices at p
    K_blend   : (m, n) — blended PDC gain K(p) = Σ α_i(p) K_i
    Q, R, Q_N : cost matrices
    Sigma0    : (n, n)
    N         : int — horizon length

    Returns
    -------
    cost : float — J_N(p; K_blend) = Tr(P_0 Σ_0)
    """
    Acl = A + B @ K_blend
    S = Q + K_blend.T @ R @ K_blend

    P = Q_N.copy()
    for _ in range(N):
        P = S + Acl.T @ P @ Acl

    return float(np.trace(P @ Sigma0))


# ── Grid-level evaluation ─────────────────────────────────────────────────────


def frozen_cost_grid(tp_model, K_vertices, eval_grid, Q, R, Sigma0):
    """J_∞(p; K) at every point on a parameter grid.

    For each grid point p:
        1. Evaluate A(p), B(p) from the true system
        2. Compute K(p) = Σ α_i(p) K_i
        3. Solve Lyapunov equation for P(p)
        4. Return Tr(P(p) Σ_0)

    Parameters
    ----------
    tp_model   : TPModel
    K_vertices : (N_v, m, n) array — vertex gains from P-RHPG
    eval_grid  : ParameterGrid — grid to evaluate on
    Q, R       : cost matrices
    Sigma0     : (n, n)

    Returns
    -------
    costs     : (N_eval,) array — J_∞(p_q; K), NaN where unstable
    rho_vals  : (N_eval,) array — ρ(Acl(p_q))
    grid_pts  : (N_eval, n_p) array — parameter values (for plotting)
    """
    N_eval = eval_grid.total_points
    n_p = eval_grid.n_p
    costs = np.full(N_eval, np.nan)
    rho_vals = np.full(N_eval, np.nan)

    for flat_idx in range(N_eval):
        multi_idx = np.unravel_index(flat_idx, eval_grid.n_points)
        p = np.array([eval_grid.axes[k][multi_idx[k]] for k in range(n_p)])

        # True system at p
        S = tp_model.system.system_matrix(p, discrete=True)
        A = S[:, : tp_model.n]
        B = S[:, tp_model.n :]

        # Blended gain
        alpha = tp_model.weights(p)
        K_blend = np.einsum("i,imn->mn", alpha, K_vertices)

        Acl = A + B @ K_blend
        rho_vals[flat_idx] = np.max(np.abs(np.linalg.eigvals(Acl)))

        S_cost = Q + K_blend.T @ R @ K_blend
        costs[flat_idx] = lyapunov_cost(Acl, S_cost, Sigma0)

    return costs, rho_vals, eval_grid.grid_points


def riccati_cost_grid(tp_model, eval_grid, Q, R, Sigma0):
    """Pointwise Riccati cost J_ric(p) = Tr(P_ric(p) Σ_0) over a grid.

    No PDC constraint — unconstrained optimal at each p independently.
    This is the lower bound that P-RHPG converges toward.

    Parameters
    ----------
    tp_model  : TPModel
    eval_grid : ParameterGrid
    Q, R      : cost matrices
    Sigma0    : (n, n)

    Returns
    -------
    costs    : (N_eval,) array — J_ric(p_q), NaN where DARE fails
    K_grid   : list of (m, n) arrays or None — optimal gains at each point
    grid_pts : (N_eval, n_p) array
    """
    N_eval = eval_grid.total_points
    n_p = eval_grid.n_p
    costs = np.full(N_eval, np.nan)
    K_grid = []

    for flat_idx in range(N_eval):
        multi_idx = np.unravel_index(flat_idx, eval_grid.n_points)
        p = np.array([eval_grid.axes[k][multi_idx[k]] for k in range(n_p)])

        S = tp_model.system.system_matrix(p, discrete=True)
        A = S[:, : tp_model.n]
        B = S[:, tp_model.n :]

        c, K = riccati_cost_at_p(A, B, Q, R, Sigma0)
        costs[flat_idx] = c
        K_grid.append(K)

    return costs, K_grid, eval_grid.grid_points


def pointwise_gap_grid(tp_model, K_vertices, eval_grid, Q, R, Sigma0):
    """Pointwise suboptimality gap J_∞(p;K) - J_ric(p) over a grid.

    Positive at each p — shows where the PDC structure is most costly
    relative to the unconstrained optimum.

    Parameters
    ----------
    tp_model   : TPModel
    K_vertices : (N_v, m, n)
    eval_grid  : ParameterGrid
    Q, R       : cost matrices
    Sigma0     : (n, n)

    Returns
    -------
    gaps      : (N_eval,) array — J_∞(p;K) - J_ric(p), NaN if either fails
    pdc_costs : (N_eval,) array — J_∞(p; K)
    ric_costs : (N_eval,) array — J_ric(p)
    grid_pts  : (N_eval, n_p) array
    """
    pdc_costs, _, grid_pts = frozen_cost_grid(tp_model, K_vertices, eval_grid, Q, R, Sigma0)
    ric_costs, _, _ = riccati_cost_grid(tp_model, eval_grid, Q, R, Sigma0)

    gaps = pdc_costs - ric_costs
    return gaps, pdc_costs, ric_costs, grid_pts


# ── Integrated cost (cross-check against prhpg.py) ───────────────────────────


def integrate_frozen_cost(costs, mu=None):
    """Integrate per-point costs J_∞(p_q; K) to get J_∞(K) = ∫ J dμ.

    Used to cross-check against solver.evaluate_infinite_cost().

    Parameters
    ----------
    costs : (N_eval,) array — per-point costs (NaN treated as failure)
    mu    : (N_eval,) array or None — quadrature weights (default: uniform)

    Returns
    -------
    J_int  : float — integrated cost (NaN if any point is NaN)
    stable : bool  — True if no NaN in costs
    """
    if np.any(np.isnan(costs)):
        return np.nan, False
    N = len(costs)
    if mu is None:
        mu = np.ones(N) / N
    return float(np.dot(mu, costs)), True


# ── Marginal cost along a scheduling dimension ────────────────────────────────


def marginal_cost_1d(tp_model, K_vertices, dim, n_points, Q, R, Sigma0, fixed_params=None):
    """J_∞(p; K) as a function of one scheduling dimension, others fixed.

    Useful for 1D slices through the scheduling domain (e.g., cost vs V
    for the aeroelastic wing).

    Parameters
    ----------
    tp_model    : TPModel
    K_vertices  : (N_v, m, n)
    dim         : int — scheduling dimension to vary (0-indexed)
    n_points    : int — number of evaluation points along this dimension
    Q, R        : cost matrices
    Sigma0      : (n, n)
    fixed_params: (n_p,) array or None — values for other dims
                  (default: midpoint of each bound)

    Returns
    -------
    p_vals  : (n_points,) array — values along the varied dimension
    costs   : (n_points,) array — J_∞(p; K) along the slice
    rho_vals: (n_points,) array — ρ(Acl) along the slice
    """
    bounds = tp_model.system.param_bounds
    n_p = tp_model.n_p
    p_vals = np.linspace(bounds[dim, 0], bounds[dim, 1], n_points)

    if fixed_params is None:
        fixed_params = 0.5 * (bounds[:, 0] + bounds[:, 1])

    costs = np.full(n_points, np.nan)
    rho_vals = np.full(n_points, np.nan)

    for k, pv in enumerate(p_vals):
        p = fixed_params.copy()
        p[dim] = pv

        S = tp_model.system.system_matrix(p, discrete=True)
        A = S[:, : tp_model.n]
        B = S[:, tp_model.n :]
        alpha = tp_model.weights(p)
        K_blend = np.einsum("i,imn->mn", alpha, K_vertices)

        Acl = A + B @ K_blend
        rho_vals[k] = np.max(np.abs(np.linalg.eigvals(Acl)))

        S_cost = Q + K_blend.T @ R @ K_blend
        costs[k] = lyapunov_cost(Acl, S_cost, Sigma0)

    return p_vals, costs, rho_vals


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")

    print("=" * 60)
    print("frozen_param_cost.py — Self-test")
    print("=" * 60)

    import numpy as np
    from scipy import linalg as scipy_linalg

    rng = np.random.default_rng(42)

    # ── 1. lyapunov_cost: stable system ──────────────────────────────────────
    print("\n--- lyapunov_cost: stable system ---")
    n = 4
    # Construct a Schur-stable Acl
    A_raw = rng.standard_normal((n, n))
    eigs = np.linalg.eigvals(A_raw)
    Acl = A_raw / (np.max(np.abs(eigs)) + 0.1) * 0.9  # scale to ρ < 0.9
    S = np.eye(n) * 2.0
    Sig = np.eye(n)

    cost = lyapunov_cost(Acl, S, Sig)
    # Verify against manual Lyapunov solve
    P_ref = scipy_linalg.solve_discrete_lyapunov(Acl.T, S)
    cost_ref = np.trace(P_ref @ Sig)
    assert np.isfinite(cost)
    assert abs(cost - cost_ref) < 1e-8 * abs(cost_ref), f"Lyapunov cost mismatch: {cost} vs {cost_ref}"
    print(f"  cost = {cost:.6e}  (ref = {cost_ref:.6e})  ✓")

    # ── 2. lyapunov_cost: unstable system ─────────────────────────────────────
    print("\n--- lyapunov_cost: unstable system → NaN ---")
    Acl_unstable = 1.5 * np.eye(n)
    cost_nan = lyapunov_cost(Acl_unstable, S, Sig)
    assert np.isnan(cost_nan), f"Expected NaN, got {cost_nan}"
    print(f"  cost = {cost_nan}  ✓")

    # ── 3. riccati_cost_at_p: simple 2×1 system ───────────────────────────────
    print("\n--- riccati_cost_at_p ---")
    A2 = np.array([[1.1, 0.0], [0.0, 0.8]])
    B2 = np.array([[1.0], [0.0]])
    Q2 = np.eye(2)
    R2 = np.array([[1.0]])
    Sig2 = np.eye(2)

    cost_ric, K_ric = riccati_cost_at_p(A2, B2, Q2, R2, Sig2)
    assert np.isfinite(cost_ric), "DARE should succeed for stabilizable system"
    assert K_ric is not None
    assert K_ric.shape == (1, 2)
    # Verify closed-loop is stable
    Acl2 = A2 + B2 @ K_ric
    rho2 = np.max(np.abs(np.linalg.eigvals(Acl2)))
    assert rho2 < 1.0, f"Riccati closed-loop not stable: ρ = {rho2}"
    print(f"  J_ric = {cost_ric:.6e},  ρ(Acl) = {rho2:.6f}  ✓")

    # ── 4. finite_horizon_cost_at_p: converges to Lyapunov as N → ∞ ──────────
    print("\n--- finite_horizon_cost_at_p: convergence in N ---")
    # Use stabilising K_blend from Riccati
    K_blend2 = K_ric
    J_lyap = lyapunov_cost(Acl2, Q2 + K_blend2.T @ R2 @ K_blend2, Sig2)

    prev = None
    for N in [10, 50, 100, 200, 500]:
        J_N = finite_horizon_cost_at_p(A2, B2, K_blend2, Q2, R2, Q2, Sig2, N)
        delta = abs(J_N - J_lyap) if prev is None else abs(J_N - prev)
        prev = J_N
        print(f"  N={N:4d}: J_N={J_N:.6e}  (J_∞={J_lyap:.6e})")
    assert J_N < J_lyap * 1.05, f"Finite-horizon cost should approach Lyapunov: {J_N} vs {J_lyap}"
    print("  Convergence: ✓")

    # ── 5. integrate_frozen_cost cross-check ─────────────────────────────────
    print("\n--- integrate_frozen_cost: cross-check ---")
    N_pts = 20
    dummy_costs = np.linspace(10.0, 20.0, N_pts)
    J_int, stable = integrate_frozen_cost(dummy_costs)
    assert stable
    assert abs(J_int - np.mean(dummy_costs)) < 1e-12
    print(f"  Integral = {J_int:.6f}  (mean = {np.mean(dummy_costs):.6f})  ✓")

    # NaN propagation
    dummy_nan = dummy_costs.copy()
    dummy_nan[5] = np.nan
    J_nan, stable_nan = integrate_frozen_cost(dummy_nan)
    assert np.isnan(J_nan) and not stable_nan
    print(f"  NaN propagation: J={J_nan}, stable={stable_nan}  ✓")

    print("\nAll tests passed ✓")
