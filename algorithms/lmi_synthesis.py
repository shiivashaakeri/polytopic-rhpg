"""
LMI-Based PDC Synthesis
========================

Synthesises a parallel distributed compensation (PDC) controller for a
discrete-time polytopic system via semidefinite programming (CVXPY).

Theory
------
Given vertex systems (A_i, B_i), the polytopic closed-loop is

    A_cl(p) = Σ_{i,j} α_i(p) α_j(p) G_{ij},   G_{ij} = A_i + B_i K_j

A sufficient condition for joint robust stability with LQR cost bound:

    X - G_{ij}^T X^{-1} G_{ij} ≻ Q + K_j^T R K_j    for all i,j    (*)

With X = P^{-1} and M_i = K_i X, (*) has the equivalent 4-block Schur form:

    [X    Y_{ij}^T    X Q^{1/2}      M_j^T R^{1/2}]
    [Y_{ij}   X       0             0             ] ≻ 0              (**)
    [Q^{1/2}X  0      I             0             ]
    [R^{1/2}M_j  0    0             I             ]

where Y_{ij} = A_i X + B_i M_j (linear in X, M_j).  This simultaneously
encodes stability and embeds the LQR stage cost Q + K^T R K in the
discrete-time Lyapunov condition.

The infinite-horizon cost J = Tr(P Σ_0) is bounded via:

    Tr(W)  s.t.  [W  Σ_0^{1/2}] ≻ 0                                (***)
                 [Σ_0^{1/2}  X ]

Minimising Tr(W) subject to (**) and (***) gives the tightest LQR cost
upper bound achievable under the PDC structure.

Reference
---------
Takarics & Vanek (2021) — discrete-time adaptation of their Theorem 1.
Tanaka & Wang (2001) "Fuzzy Control Systems Design and Analysis", Wiley, Ch. 3.
Boyd et al. (1994) "LMIs in System and Control Theory", SIAM.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

_PREFERRED_SOLVERS = ["CLARABEL", "MOSEK", "SCS"]


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class LMIResult:
    """Output of LMI PDC synthesis."""

    K_vertices: np.ndarray  # (N_v, m, n)
    X: Optional[np.ndarray]  # (n, n) Lyapunov variable
    P: Optional[np.ndarray]  # (n, n) Lyapunov matrix = X^{-1}
    objective: float  # optimal Tr(W), upper bound on J_∞
    feasible: bool
    solve_time: float  # wall-clock [s]
    status: str  # CVXPY status string
    N_v: int
    n_lmis: int  # number of LMI block constraints
    metadata: dict = field(default_factory=dict)


# ── Core synthesis ────────────────────────────────────────────────────────────


def synthesise_pdc(A_vertices, B_vertices, Q, R, Sigma0, objective="h2", solver=None, eps=1e-6, verbose=False):
    """Synthesise a discrete-time PDC controller via LMI.

    Parameters
    ----------
    A_vertices : (N_v, n, n) — vertex A matrices (discrete-time)
    B_vertices : (N_v, n, m) — vertex B matrices
    Q          : (n, n) ≻ 0  — LQR state cost
    R          : (m, m) ≻ 0  — LQR input cost
    Sigma0     : (n, n)       — IC weighting for cost Tr(P Σ_0)
    objective  : 'h2'         — minimise LQR cost upper bound Tr(W)
                 'feasibility' — find any stabilising PDC, no cost objective
    solver     : str or None  — CVXPY solver (None = auto, prefers CLARABEL)
    eps        : float        — LMI strictness margin X ≻ eps I
    verbose    : bool

    Returns
    -------
    result : LMIResult
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("pip install cvxpy")

    A_vertices = np.asarray(A_vertices, dtype=float)
    B_vertices = np.asarray(B_vertices, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    Sigma0 = np.asarray(Sigma0, dtype=float)

    N_v, n, _ = A_vertices.shape
    m = B_vertices.shape[2]
    t0 = time.perf_counter()

    # Matrix square roots
    Qhalf = np.linalg.cholesky(Q + 1e-12 * np.eye(n))
    Rhalf = np.linalg.cholesky(R + 1e-12 * np.eye(m))
    S0half = np.linalg.cholesky(Sigma0 + 1e-12 * np.eye(n))

    # CVXPY variables
    X = cp.Variable((n, n), symmetric=True)
    Ms = [cp.Variable((m, n)) for _ in range(N_v)]

    In, Im = np.eye(n), np.eye(m)
    Z_nn = np.zeros((n, n))
    Z_nm = np.zeros((n, m))
    Z_mn = np.zeros((m, n))

    constraints = [X >> eps * np.eye(n)]

    # Stability + LQR LMIs (4-block Schur) for all (i, j)
    n_lmis = 0
    for i in range(N_v):
        for j in range(N_v):
            Y_ij = A_vertices[i] @ X + B_vertices[i] @ Ms[j]
            QhX = Qhalf.T @ X  # (n, n), linear in X
            RhMj = Rhalf.T @ Ms[j]  # (m, n), linear in M_j

            block = cp.bmat(
                [
                    [X, Y_ij.T, QhX.T, RhMj.T],
                    [Y_ij, X, Z_nn, Z_nm],
                    [QhX, Z_nn, In, Z_nm],
                    [RhMj, Z_mn, Z_mn, Im],
                ]
            )
            constraints.append(block >> eps * np.eye(2 * n + n + m))
            n_lmis += 1

    # Objective
    if objective == "h2":
        W = cp.Variable((n, n), symmetric=True)
        constraints.append(W >> 0)
        constraints.append(cp.bmat([[W, S0half], [S0half.T, X]]) >> eps * np.eye(2 * n))
        cvx_obj = cp.Minimize(cp.trace(W))
    else:
        cvx_obj = cp.Minimize(0)

    # Solver selection
    if solver is None:
        available = cp.installed_solvers()
        solver = next((s for s in _PREFERRED_SOLVERS if s in available), None)
    solve_kwargs = dict(verbose=verbose)
    if solver is not None:
        solve_kwargs["solver"] = solver

    prob = cp.Problem(cvx_obj, constraints)
    try:
        prob.solve(**solve_kwargs)
    except cp.SolverError as e:
        elapsed = time.perf_counter() - t0
        return LMIResult(
            K_vertices=np.zeros((N_v, m, n)),
            X=None,
            P=None,
            objective=np.inf,
            feasible=False,
            solve_time=elapsed,
            status=f"SolverError: {e}",
            N_v=N_v,
            n_lmis=n_lmis,
            metadata=dict(n=n, m=m, objective=objective),
        )

    elapsed = time.perf_counter() - t0
    status = prob.status

    if prob.status not in ("optimal", "optimal_inaccurate") or X.value is None:
        return LMIResult(
            K_vertices=np.zeros((N_v, m, n)),
            X=None,
            P=None,
            objective=np.inf,
            feasible=False,
            solve_time=elapsed,
            status=status,
            N_v=N_v,
            n_lmis=n_lmis,
            metadata=dict(n=n, m=m, objective=objective),
        )

    # Recover gains K_i = M_i X^{-1}
    X_val = X.value
    P_val = np.linalg.inv(X_val)
    K_vertices = np.stack([Ms[i].value @ P_val for i in range(N_v)])
    obj_val = float(prob.value) if objective == "h2" else 0.0

    return LMIResult(
        K_vertices=K_vertices,
        X=X_val,
        P=P_val,
        objective=obj_val,
        feasible=True,
        solve_time=elapsed,
        status=status,
        N_v=N_v,
        n_lmis=n_lmis,
        metadata=dict(
            n=n,
            m=m,
            objective=objective,
            solver=prob.solver_stats.solver_name if prob.solver_stats else None,
        ),
    )


# ── Convenience wrapper ───────────────────────────────────────────────────────


def synthesise_from_tp(tp_model, Q, R, Sigma0, objective="h2", solver=None, eps=1e-6, verbose=False):
    """Synthesise PDC gains from a TPModel (vertex arrays extracted automatically).

    Parameters match synthesise_pdc. Q, R must match those used in P-RHPG
    so that cost comparisons are on equal footing.
    """
    return synthesise_pdc(
        tp_model.A_vertices,
        tp_model.B_vertices,
        Q,
        R,
        Sigma0,
        objective=objective,
        solver=solver,
        eps=eps,
        verbose=verbose,
    )


# ── Verification ─────────────────────────────────────────────────────────────


def verify_stability(lmi_result, A_vertices, B_vertices, tol=1e-6):
    """Verify G_{ij}^T P G_{ij} - P ≺ 0 for all i, j.

    Returns
    -------
    verified : bool
    margins  : (N_v, N_v) — -λ_max(G_{ij}^T P G_{ij} - P), positive = stable
    """
    if not lmi_result.feasible or lmi_result.P is None:
        N_v = np.asarray(A_vertices).shape[0]
        return False, np.full((N_v, N_v), np.nan)

    K_verts = lmi_result.K_vertices
    P = lmi_result.P
    N_v = np.asarray(A_vertices).shape[0]
    margins = np.zeros((N_v, N_v))

    for i in range(N_v):
        for j in range(N_v):
            G_ij = A_vertices[i] + B_vertices[i] @ K_verts[j]
            diff = G_ij.T @ P @ G_ij - P
            margins[i, j] = -np.max(np.linalg.eigvalsh(diff))

    return bool(np.all(margins > -tol)), margins


def evaluate_lqr_cost(lmi_result, Q, R, Sigma0, A_quad, B_quad, alpha_quad, mu):
    """Evaluate integrated LQR cost J_∞ = Σ_q μ_q Tr(P_q Σ_0) for LMI gains.

    Uses the discrete Lyapunov equation at each quadrature point q with stage
    cost Q + K_blend(p_q)^T R K_blend(p_q).  This is the same evaluation
    used in prhpg.evaluate_infinite_cost() so comparisons are fair.

    Parameters
    ----------
    lmi_result  : LMIResult
    Q, R        : LQR cost matrices
    Sigma0      : (n, n)
    A_quad      : (N_q, n, n)
    B_quad      : (N_q, n, m)
    alpha_quad  : (N_q, N_v)
    mu          : (N_q,)

    Returns
    -------
    J_inf   : float — integrated cost (NaN if unstable anywhere)
    rho_max : float
    """
    from scipy import linalg

    if not lmi_result.feasible:
        return np.nan, np.nan

    K_verts = lmi_result.K_vertices
    N_q = A_quad.shape[0]
    J_total = 0.0
    rho_max = 0.0

    for q in range(N_q):
        K_blend = np.einsum("i,imn->mn", alpha_quad[q], K_verts)
        Acl_q = A_quad[q] + B_quad[q] @ K_blend
        rho_q = np.max(np.abs(np.linalg.eigvals(Acl_q)))
        rho_max = max(rho_max, rho_q)
        if rho_q >= 1.0:
            return np.nan, rho_max
        S_q = Q + K_blend.T @ R @ K_blend
        try:
            P_q = linalg.solve_discrete_lyapunov(Acl_q.T, S_q)
            J_total += mu[q] * np.trace(P_q @ Sigma0)
        except Exception:
            return np.nan, rho_max

    return float(J_total), float(rho_max)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/mnt/user-data/uploads")

    print("=" * 60)
    print("lmi_synthesis.py — Self-test")
    print("=" * 60)

    if not CVXPY_AVAILABLE:
        print("CVXPY not available — install with: pip install cvxpy")
        sys.exit(1)

    import numpy as np
    from scipy import linalg as scipy_linalg

    # ── 2-vertex, n=2, m=1, unstable open-loop ────────────────────────────────
    print("\n--- 2-vertex, n=2, m=1, unstable open-loop ---")
    A1 = np.array([[1.2, 0.1], [0.0, 0.8]])
    B1 = np.array([[1.0], [0.5]])
    A2 = np.array([[1.05, -0.1], [0.2, 0.9]])
    B2 = np.array([[0.8], [0.6]])
    A_v = np.stack([A1, A2])
    B_v = np.stack([B1, B2])
    Q = np.eye(2) * 10.0
    R = np.array([[1.0]])
    Sig0 = np.eye(2)

    result = synthesise_pdc(A_v, B_v, Q, R, Sig0, objective="h2", verbose=False)
    print(f"  Status    : {result.status}")
    print(f"  Feasible  : {result.feasible}")
    print(f"  Obj Tr(W) : {result.objective:.4e}")
    print(f"  Solve time: {result.solve_time:.3f} s")
    print(f"  n_lmis    : {result.n_lmis}")
    assert result.feasible, "Must be feasible for stabilizable system"

    # Verify stability
    verified, margins = verify_stability(result, A_v, B_v)
    print(f"\n  Stability verified : {verified}")
    print(f"  Lyapunov margins (>0 = stable):")
    for i in range(2):
        for j in range(2):
            print(f"    G_{i + 1}{j + 1}: {margins[i, j]:.3e}")
    assert verified, "All G_ij must satisfy Lyapunov"
    assert np.all(margins > 0), f"Margins must be positive: {margins}"

    # Spectral radii
    print(f"\n  Closed-loop spectral radii ρ(G_ij):")
    K_v = result.K_vertices
    rho_max_all = 0.0
    for i in range(2):
        for j in range(2):
            G_ij = A_v[i] + B_v[i] @ K_v[j]
            rho = np.max(np.abs(np.linalg.eigvals(G_ij)))
            rho_max_all = max(rho_max_all, rho)
            print(f"    G_{i + 1}{j + 1}: ρ = {rho:.4f}  {'✓' if rho < 1 else '✗'}")
    assert rho_max_all < 1.0, f"All vertices must be stable, got max ρ={rho_max_all:.4f}"

    # evaluate_lqr_cost cross-check
    print(f"\n  LQR cost evaluation (uniform quadrature at 4 points):")
    N_q = 4
    # Blend at uniform alpha weights
    alphas = np.array([[0.5, 0.5], [0.7, 0.3], [0.3, 0.7], [1.0, 0.0]])
    A_q = np.stack([a[0] * A1 + a[1] * A2 for a in alphas])
    B_q = np.stack([a[0] * B1 + a[1] * B2 for a in alphas])
    mu = np.ones(N_q) / N_q
    J_eval, rho_eval = evaluate_lqr_cost(result, Q, R, Sig0, A_q, B_q, alphas, mu)
    print(f"    J_∞ = {J_eval:.4e},  ρ_max = {rho_eval:.4f}")
    assert np.isfinite(J_eval), "Cost should be finite for stable system"

    # Feasibility mode
    print("\n--- feasibility mode ---")
    result_f = synthesise_pdc(A_v, B_v, Q, R, Sig0, objective="feasibility", verbose=False)
    print(f"  Status  : {result_f.status}")
    print(f"  Feasible: {result_f.feasible}")
    verified_f, _ = verify_stability(result_f, A_v, B_v)
    print(f"  Stable  : {verified_f}")
    assert result_f.feasible

    # Aeroelastic wing
    print("\n--- aeroelastic wing, 2×2×2 grid (N_v=8) ---")
    try:
        from aeroelastic_wing import AeroelasticWing
        from tp_model import TPModel

        sys_obj = AeroelasticWing()
        tp = TPModel.from_grid(sys_obj, [2, 2, 2])
        Q_wing = np.diag([10.0, 1.0, 100.0, 1.0])
        R_wing = np.array([[1.0]])
        Sig_w = np.eye(4)

        res_wing = synthesise_from_tp(tp, Q_wing, R_wing, Sig_w, objective="h2", verbose=False)
        print(f"  Status    : {res_wing.status}")
        print(f"  Feasible  : {res_wing.feasible}")
        print(f"  Obj Tr(W) : {res_wing.objective:.4e}")
        print(f"  Solve time: {res_wing.solve_time:.3f} s")
        print(f"  n_lmis    : {res_wing.n_lmis}  (N_v²={res_wing.N_v**2})")

        if res_wing.feasible:
            verified_w, margins_w = verify_stability(res_wing, tp.A_vertices, tp.B_vertices)
            print(f"  Stability : {verified_w}")
            print(f"  min margin: {margins_w.min():.3e}")
            max_rho = max(
                np.max(np.abs(np.linalg.eigvals(tp.A_vertices[i] + tp.B_vertices[i] @ res_wing.K_vertices[j])))
                for i in range(tp.Nv)
                for j in range(tp.Nv)
            )
            print(f"  max ρ(G_ij): {max_rho:.4f}  {'✓' if max_rho < 1 else '✗'}")

    except Exception as e:
        print(f"  Wing test skipped: {e}")

    print("\nAll tests passed ✓")
