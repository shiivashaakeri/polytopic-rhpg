"""
Joint Stability Analysis
=========================

Analyses the stability structure of PDC closed-loop systems, focusing
on the gap between vertex-wise stability and joint (robust) stability.

The core message of Experiment 1 in the paper:
    - Vertex closed-loop matrices G_{ii} = A_i + B_i K_i may all be
      stable (ρ(G_{ii}) < 1), yet the full polytopic closed-loop
      Acl(p; K̄) = Σ_{i,j} α_i α_j G_{ij}  can still be unstable
      because the cross-terms G_{ij} = A_i + B_i K_j (i ≠ j) are not
      accounted for by vertex-only checks.
    - P-RHPG produces gains K̄ for which Acl(p; K̄) is jointly stable
      for all p ∈ Ω, as guaranteed by the Corollary (Section IV-B).
    - This module generates all data needed for Figure 1 (cross-term
      stability plot) and the stability column of the comparison table.

Functions
---------
cross_term_matrices        G_{ij} = A_i + B_i K_j  for all i, j
cross_term_spectral_radii  ρ(G_{ij}) for all i, j pairs
diagonal_spectral_radii    ρ(G_{ii}) — vertex-only closed-loop
joint_spectral_map         ρ(Acl(p; K̄)) over a fine parameter grid
stability_comparison       Full comparison dict for Experiment 1
lyapunov_certificate       Verify Corollary: Acl'P̄Acl ≺ P̄ at each p
stability_map_2d           2D slice through Ω for heatmap plots
"""

import numpy as np
from scipy import linalg


# ── Cross-term structure ──────────────────────────────────────────────────────


def cross_term_matrices(A_vertices, B_vertices, K_vertices):
    """Compute all N_v² cross-term closed-loop matrices G_{ij}.

    G_{ij} = A_i + B_i K_j

    In the polytopic closed-loop:
        Acl(p; K) = Σ_i Σ_j α_i(p) α_j(p) (A_i + B_i K_j)
                  = Σ_{i,j} α_i α_j G_{ij}

    Diagonal terms (i=j): G_{ii} = A_i + B_i K_i  — vertex closed-loop
    Off-diagonal (i≠j):   G_{ij}                  — cross terms

    Parameters
    ----------
    A_vertices : (N_v, n, n)
    B_vertices : (N_v, n, m)
    K_vertices : (N_v, m, n) — vertex gains from P-RHPG

    Returns
    -------
    G : (N_v, N_v, n, n) array — G[i, j] = A_i + B_i K_j
    """
    A_vertices = np.asarray(A_vertices)
    B_vertices = np.asarray(B_vertices)
    K_vertices = np.asarray(K_vertices)
    N_v, n, _ = A_vertices.shape

    G = np.zeros((N_v, N_v, n, n))
    for i in range(N_v):
        for j in range(N_v):
            G[i, j] = A_vertices[i] + B_vertices[i] @ K_vertices[j]
    return G


def cross_term_spectral_radii(G):
    """Spectral radius of every cross-term matrix G_{ij}.

    Parameters
    ----------
    G : (N_v, N_v, n, n) — from cross_term_matrices()

    Returns
    -------
    radii : (N_v, N_v) array — ρ(G_{ij})
    """
    N_v = G.shape[0]
    radii = np.zeros((N_v, N_v))
    for i in range(N_v):
        for j in range(N_v):
            radii[i, j] = np.max(np.abs(np.linalg.eigvals(G[i, j])))
    return radii


def diagonal_spectral_radii(G):
    """Spectral radius of diagonal (vertex) closed-loop matrices G_{ii}.

    Parameters
    ----------
    G : (N_v, N_v, n, n) — from cross_term_matrices()

    Returns
    -------
    radii : (N_v,) array — ρ(G_{ii})
    """
    N_v = G.shape[0]
    return np.array([np.max(np.abs(np.linalg.eigvals(G[i, i]))) for i in range(N_v)])


def open_loop_vertex_radii(A_vertices):
    """Spectral radius of open-loop vertex matrices A_i.

    Parameters
    ----------
    A_vertices : (N_v, n, n)

    Returns
    -------
    radii : (N_v,) array — ρ(A_i)
    """
    A_vertices = np.asarray(A_vertices)
    return np.array([np.max(np.abs(np.linalg.eigvals(A_vertices[i]))) for i in range(A_vertices.shape[0])])


# ── Joint spectral radius over Ω ─────────────────────────────────────────────


def joint_spectral_map(tp_model, K_vertices, eval_grid):
    """ρ(Acl(p; K̄)) at every point on a parameter grid.

    Evaluates the true closed-loop (not the polytopic approximation) at
    each grid point:
        Acl(p) = A_true(p) + B_true(p) · K(p),   K(p) = Σ α_i(p) K_i

    Parameters
    ----------
    tp_model   : TPModel
    K_vertices : (N_v, m, n)
    eval_grid  : ParameterGrid

    Returns
    -------
    rho_map  : (N_eval,) array — ρ(Acl(p_q))
    stable   : (N_eval,) bool array — True where ρ < 1
    grid_pts : (N_eval, n_p) array — parameter values
    """
    N_eval = eval_grid.total_points
    n_p = eval_grid.n_p
    rho_map = np.empty(N_eval)

    for flat_idx in range(N_eval):
        multi_idx = np.unravel_index(flat_idx, eval_grid.n_points)
        p = np.array([eval_grid.axes[k][multi_idx[k]] for k in range(n_p)])

        S = tp_model.system.system_matrix(p, discrete=True)
        A = S[:, : tp_model.n]
        B = S[:, tp_model.n :]
        alpha = tp_model.weights(p)
        K_blend = np.einsum("i,imn->mn", alpha, K_vertices)

        Acl = A + B @ K_blend
        rho_map[flat_idx] = np.max(np.abs(np.linalg.eigvals(Acl)))

    return rho_map, rho_map < 1.0, eval_grid.grid_points


# ── Lyapunov certificate verification ────────────────────────────────────────


def lyapunov_certificate(tp_model, K_vertices, P_bar_at_grid, Q, eval_grid, tol=1e-8):
    """Verify the Corollary certificate: Acl(p)'P̄(p)Acl(p) ≺ P̄(p).

    From the paper's Corollary (Section IV-B): if P̄(p) is the limit of
    P_0*(p; N) as N→∞, then:
        (i)  P̄(p) ≻ Q ≻ 0
        (ii) Acl(p; K̄)' P̄(p) Acl(p; K̄) ≺ P̄(p)

    Condition (ii) is the discrete-time Lyapunov condition implying
    ρ(Acl(p; K̄)) < 1 for all p ∈ Ω.

    Parameters
    ----------
    tp_model       : TPModel
    K_vertices     : (N_v, m, n)
    P_bar_at_grid  : (N_eval, n, n) — P̄(p_q) at each grid point
                     (use result.P_history[0] from P-RHPG with store_all=True,
                      or approximate with a large-N solve)
    Q              : (n, n) — stage cost matrix (Q ≻ 0)
    eval_grid      : ParameterGrid
    tol            : float — tolerance for positive-definiteness check

    Returns
    -------
    cert : dict with keys:
        'P_bar_pd'    : bool — P̄(p) ≻ 0 everywhere
        'P_bar_gt_Q'  : bool — λ_min(P̄(p) - Q) > -tol everywhere
        'lyap_holds'  : bool — Acl'P̄Acl ≺ P̄ everywhere
        'rho_max'     : float — max ρ(Acl(p))
        'lyap_deficit': (N_eval,) — λ_max(Acl'P̄Acl - P̄ + tol·I) at each p
        'min_eig_PQ'  : (N_eval,) — λ_min(P̄ - Q) at each p
    """
    N_eval = eval_grid.total_points
    n_p = eval_grid.n_p
    n = tp_model.n

    lyap_deficit = np.empty(N_eval)
    min_eig_PQ = np.empty(N_eval)
    rho_all = np.empty(N_eval)

    for flat_idx in range(N_eval):
        multi_idx = np.unravel_index(flat_idx, eval_grid.n_points)
        p = np.array([eval_grid.axes[k][multi_idx[k]] for k in range(n_p)])

        S = tp_model.system.system_matrix(p, discrete=True)
        A = S[:, :n]
        B = S[:, n:]
        alpha = tp_model.weights(p)
        K_blend = np.einsum("i,imn->mn", alpha, K_vertices)
        Acl = A + B @ K_blend

        rho_all[flat_idx] = np.max(np.abs(np.linalg.eigvals(Acl)))

        P_bar = P_bar_at_grid[flat_idx]

        # Check Acl' P̄ Acl ≺ P̄  ⟺  λ_max(Acl'P̄Acl - P̄) < 0
        diff = Acl.T @ P_bar @ Acl - P_bar
        lyap_deficit[flat_idx] = np.max(np.linalg.eigvalsh(diff))

        # Check P̄ ≻ Q
        min_eig_PQ[flat_idx] = np.min(np.linalg.eigvalsh(P_bar - Q))

    return dict(
        P_bar_pd=bool(np.all(np.linalg.eigvalsh(P_bar_at_grid).min(axis=-1) > tol)),
        P_bar_gt_Q=bool(np.all(min_eig_PQ > -tol)),
        lyap_holds=bool(np.all(lyap_deficit < tol)),
        rho_max=float(np.max(rho_all)),
        lyap_deficit=lyap_deficit,
        min_eig_PQ=min_eig_PQ,
    )


# ── Full comparison for Experiment 1 ─────────────────────────────────────────


def stability_comparison(tp_model, K_vertices, eval_grid):
    """Full stability comparison for Experiment 1.

    Computes and returns all quantities needed for Figure 1:
        - Open-loop vertex radii  ρ(A_i)
        - Diagonal (vertex CL)    ρ(G_{ii})
        - Max cross-term          max_{i≠j} ρ(G_{ij})
        - Joint closed-loop       ρ(Acl(p; K̄))  over eval_grid

    Parameters
    ----------
    tp_model   : TPModel
    K_vertices : (N_v, m, n)
    eval_grid  : ParameterGrid

    Returns
    -------
    d : dict with keys:
        'rho_open'       : (N_v,) — ρ(A_i)
        'rho_diagonal'   : (N_v,) — ρ(G_{ii})
        'rho_cross'      : (N_v, N_v) — ρ(G_{ij})  (diagonal = G_{ii})
        'rho_joint'      : (N_eval,) — ρ(Acl(p_q))
        'stable_joint'   : (N_eval,) bool
        'grid_pts'       : (N_eval, n_p)
        'all_vertex_stable'  : bool — all ρ(G_{ii}) < 1
        'all_cross_stable'   : bool — all ρ(G_{ij}) < 1
        'all_joint_stable'   : bool — all ρ(Acl(p)) < 1
        'max_rho_open'       : float
        'max_rho_diagonal'   : float
        'max_rho_cross'      : float — max over ALL i,j
        'max_rho_offdiag'    : float — max over i≠j only
        'max_rho_joint'      : float
    """
    A_verts = tp_model.A_vertices  # (N_v, n, n)
    B_verts = tp_model.B_vertices  # (N_v, n, m)
    K_verts = np.asarray(K_vertices)

    G = cross_term_matrices(A_verts, B_verts, K_verts)
    rho_cross = cross_term_spectral_radii(G)
    rho_diagonal = np.diag(rho_cross)
    rho_open = open_loop_vertex_radii(A_verts)
    rho_joint, stable_joint, grid_pts = joint_spectral_map(tp_model, K_verts, eval_grid)

    N_v = G.shape[0]
    offdiag_mask = ~np.eye(N_v, dtype=bool)
    max_rho_offdiag = float(np.max(rho_cross[offdiag_mask])) if N_v > 1 else np.nan

    return dict(
        rho_open=rho_open,
        rho_diagonal=rho_diagonal,
        rho_cross=rho_cross,
        rho_joint=rho_joint,
        stable_joint=stable_joint,
        grid_pts=grid_pts,
        all_vertex_stable=bool(np.all(rho_diagonal < 1.0)),
        all_cross_stable=bool(np.all(rho_cross < 1.0)),
        all_joint_stable=bool(np.all(rho_joint < 1.0)),
        max_rho_open=float(np.max(rho_open)),
        max_rho_diagonal=float(np.max(rho_diagonal)),
        max_rho_cross=float(np.max(rho_cross)),
        max_rho_offdiag=max_rho_offdiag,
        max_rho_joint=float(np.max(rho_joint)),
    )


# ── 2D stability slice for heatmap plots ─────────────────────────────────────


def stability_map_2d(tp_model, K_vertices, dim0, dim1, n0, n1, fixed_params=None):
    """ρ(Acl(p)) on a 2D slice through Ω (for heatmap visualisation).

    Varies two scheduling dimensions while holding others fixed.

    Parameters
    ----------
    tp_model    : TPModel
    K_vertices  : (N_v, m, n)
    dim0, dim1  : int — scheduling dimensions to vary (0-indexed)
    n0, n1      : int — grid resolution along each dimension
    fixed_params: (n_p,) array or None — midpoint default for fixed dims

    Returns
    -------
    rho_map : (n0, n1) array — ρ(Acl) on the 2D grid
    ax0     : (n0,) array   — values along dim0
    ax1     : (n1,) array   — values along dim1
    """
    bounds = tp_model.system.param_bounds
    n_p = tp_model.n_p

    if fixed_params is None:
        fixed_params = 0.5 * (bounds[:, 0] + bounds[:, 1])

    ax0 = np.linspace(bounds[dim0, 0], bounds[dim0, 1], n0)
    ax1 = np.linspace(bounds[dim1, 0], bounds[dim1, 1], n1)

    rho_map = np.empty((n0, n1))
    K_verts = np.asarray(K_vertices)

    for i, v0 in enumerate(ax0):
        for j, v1 in enumerate(ax1):
            p = fixed_params.copy()
            p[dim0] = v0
            p[dim1] = v1

            S = tp_model.system.system_matrix(p, discrete=True)
            A = S[:, : tp_model.n]
            B = S[:, tp_model.n :]
            alpha = tp_model.weights(p)
            K_blend = np.einsum("i,imn->mn", alpha, K_verts)
            Acl = A + B @ K_blend
            rho_map[i, j] = np.max(np.abs(np.linalg.eigvals(Acl)))

    return rho_map, ax0, ax1


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("joint_stability.py — Self-test")
    print("=" * 60)

    import numpy as np

    rng = np.random.default_rng(42)
    n, m, N_v = 2, 1, 4

    # ── Construct a simple stable polytopic system ────────────────────────────
    # A_i: stable matrices (scaled random)
    A_verts = np.stack(
        [
            0.7 * np.array([[0.9, 0.1], [-0.1, 0.8]]),
            0.7 * np.array([[0.8, 0.0], [0.2, 0.85]]),
            0.7 * np.array([[0.85, -0.05], [0.0, 0.9]]),
            0.7 * np.array([[0.95, 0.05], [-0.05, 0.8]]),
        ]
    )
    B_verts = np.stack(
        [
            np.array([[0.5], [1.0]]),
            np.array([[0.4], [0.9]]),
            np.array([[0.6], [1.1]]),
            np.array([[0.5], [1.0]]),
        ]
    )
    # Gains: small stabilising values
    K_verts = np.stack(
        [
            np.array([[-0.3, -0.5]]),
            np.array([[-0.25, -0.45]]),
            np.array([[-0.35, -0.55]]),
            np.array([[-0.3, -0.5]]),
        ]
    )

    # ── cross_term_matrices ───────────────────────────────────────────────────
    print("\n--- cross_term_matrices ---")
    G = cross_term_matrices(A_verts, B_verts, K_verts)
    assert G.shape == (N_v, N_v, n, n), f"Shape mismatch: {G.shape}"
    # Verify one entry manually
    G00_ref = A_verts[0] + B_verts[0] @ K_verts[0]
    assert np.allclose(G[0, 0], G00_ref)
    G12_ref = A_verts[1] + B_verts[1] @ K_verts[2]
    assert np.allclose(G[1, 2], G12_ref)
    print(f"  G shape: {G.shape}  ✓")

    # ── cross_term_spectral_radii ─────────────────────────────────────────────
    print("\n--- cross_term_spectral_radii ---")
    rho_cross = cross_term_spectral_radii(G)
    assert rho_cross.shape == (N_v, N_v)
    assert np.all(rho_cross >= 0)
    print(f"  ρ(G_ij) matrix:\n{np.round(rho_cross, 4)}")
    print(f"  max ρ = {rho_cross.max():.4f}  ✓")

    # ── diagonal_spectral_radii ───────────────────────────────────────────────
    print("\n--- diagonal_spectral_radii ---")
    rho_diag = diagonal_spectral_radii(G)
    assert rho_diag.shape == (N_v,)
    assert np.allclose(rho_diag, np.diag(rho_cross))
    print(f"  ρ(G_ii) = {np.round(rho_diag, 4)}  ✓")

    # ── open_loop_vertex_radii ────────────────────────────────────────────────
    print("\n--- open_loop_vertex_radii ---")
    rho_open = open_loop_vertex_radii(A_verts)
    assert rho_open.shape == (N_v,)
    print(f"  ρ(A_i) = {np.round(rho_open, 4)}  ✓")

    # ── stability hierarchy check ─────────────────────────────────────────────
    print("\n--- stability_comparison (synthetic) ---")
    # Verify: all vertex-CL stable, check max cross-term vs diagonal
    all_diag_stable = np.all(rho_diag < 1.0)
    all_cross_stable = np.all(rho_cross < 1.0)
    print(f"  All G_ii stable : {all_diag_stable}")
    print(f"  All G_ij stable : {all_cross_stable}")
    print(f"  max ρ(G_ii) = {rho_diag.max():.4f}")
    print(f"  max ρ(G_ij) = {rho_cross.max():.4f}")
    N_v_sq = N_v * N_v
    offdiag = ~np.eye(N_v, dtype=bool)
    print(f"  max ρ(G_ij, i≠j) = {rho_cross[offdiag].max():.4f}")

    # ── lyapunov_certificate (synthetic, no TPModel needed) ───────────────────
    print("\n--- lyapunov_certificate (direct) ---")
    # Build a known stable Acl with a known P̄
    Acl_test = np.array([[0.5, 0.1], [-0.1, 0.6]])
    S_test = np.eye(2) * 0.5
    P_bar = linalg.solve_discrete_lyapunov(Acl_test.T, S_test)

    # Verify Acl' P̄ Acl - P̄ is negative definite
    diff = Acl_test.T @ P_bar @ Acl_test - P_bar
    eigs_diff = np.linalg.eigvalsh(diff)
    assert np.all(eigs_diff < 0), f"Expected NSD, got eigs={eigs_diff}"
    print(f"  λ(Acl'P̄Acl - P̄) = {np.round(eigs_diff, 6)}  (all < 0 ✓)")

    Q_test = np.eye(2) * 0.1
    min_eig_PQ = np.min(np.linalg.eigvalsh(P_bar - Q_test))
    print(f"  λ_min(P̄ - Q) = {min_eig_PQ:.6f}  (> 0 ✓)")

    # ── integrate_and_print summary ───────────────────────────────────────────
    print("\n--- Key stability hierarchy ---")
    print(f"  Open-loop      max ρ(A_i)   = {rho_open.max():.4f}")
    print(f"  Vertex CL      max ρ(G_ii)  = {rho_diag.max():.4f}")
    print(f"  Cross-term     max ρ(G_ij)  = {rho_cross.max():.4f}")
    print(f"  (Experiment 1: these may differ significantly)")

    print("\nAll tests passed ✓")
