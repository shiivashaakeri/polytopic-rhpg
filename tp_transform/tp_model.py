"""
TP Model — Unified Polytopic Model Interface
=============================================

Wraps the full pipeline:
    nonlinear system → qLPV → sampling → HOSVD analysis → polytopic model

Provides everything P-RHPG needs:
    - Vertex systems (A_i, B_i)
    - Weight evaluation α_i(p) at arbitrary parameter points
    - Gram matrix and conditioning diagnostics
    - Reconstruction error validation
    - Nonlinear simulation for closed-loop verification

Usage:
    from tp_model import TPModel
    tp = TPModel.from_system(system, n_sample=[20, 5, 5], threshold=0.01)
    tp.summary()
    vertices = tp.vertices        # list of (A_i, B_i)
    alphas   = tp.weights(p)      # (N_v,) array
    S_p      = tp.evaluate(p)     # (n, n+m) system matrix
"""

import numpy as np
from tp_transform.sampling import ParameterGrid, build_system_tensor
from tp_transform.hosvd import hosvd as run_hosvd
from tp_transform.convex_hull import (
    multilinear_polytopic,
    multilinear_error,
    snnn_normalize,
    snnn_column_shift,
    _build_khatri_rao,
)


class TPModel:
    """Polytopic LPV model for use with P-RHPG.

    Attributes
    ----------
    system : object
        The original system (InvertedPendulum, AeroelasticWing, etc.).
    snnn : SNNNResult
        The underlying polytopic model with SNNN weights.
    hosvd_result : HOSVDResult or None
        HOSVD decomposition (for diagnostics / paper figures).
    n : int
        State dimension.
    m : int
        Input dimension.
    n_p : int
        Number of scheduling parameters.
    n_vertices : int
        Total vertex count N_v.
    vertex_grid : ParameterGrid
        Grid defining the vertex locations.
    """

    def __init__(self, system, snnn, hosvd_result=None):
        self.system = system
        self.snnn = snnn
        self.hosvd_result = hosvd_result
        self.n = snnn.n
        self.m = snnn.n_plus_m - snnn.n
        self.n_p = len(snnn.ranks)
        self.n_vertices = snnn.n_vertices
        self.vertex_grid = snnn.grid

    # ── Construction ─────────────────────────────────────────

    @classmethod
    def from_system(cls, system, n_sample, threshold=0.01, extra=1, method="multilinear", verbose=True):
        """Build a TPModel from a system object.

        Parameters
        ----------
        system : object
            Must have .system_matrix(p, discrete=True), .param_bounds.
        n_sample : list of int
            Number of sampling points per scheduling dimension for HOSVD.
        threshold : float
            HOSVD singular value cutoff: keep σ_i if σ_i/σ_1 > threshold.
        extra : int
            Extra grid points per dim beyond HOSVD rank (default 1).
        method : str
            'multilinear' (recommended) or 'simplex' or 'column_shift'.
        verbose : bool

        Returns
        -------
        tp : TPModel
        """
        sample_grid = ParameterGrid(system.param_bounds, n_points=n_sample)
        T = build_system_tensor(system, sample_grid)
        h = run_hosvd(T, sample_grid, threshold=threshold)

        if verbose:
            print("HOSVD rank analysis:")
            h.summary()

        if method == "multilinear":
            vertex_n = [r + extra for r in h.ranks]
            vertex_grid = ParameterGrid(system.param_bounds, n_points=vertex_n)
            snnn = multilinear_polytopic(system, vertex_grid)
        elif method == "simplex":
            snnn = snnn_normalize(h, T)
        elif method == "column_shift":
            snnn = snnn_column_shift(h, T)
        else:
            raise ValueError(f"Unknown method: {method}")

        tp = cls(system, snnn, hosvd_result=h)

        if verbose:
            tp.summary()

        return tp

    @classmethod
    def from_grid(cls, system, vertex_n_points):
        """Build directly from explicit vertex grid (no HOSVD).

        Parameters
        ----------
        system : object
        vertex_n_points : list of int — grid points per dimension
        """
        vertex_grid = ParameterGrid(system.param_bounds, n_points=vertex_n_points)
        snnn = multilinear_polytopic(system, vertex_grid)
        return cls(system, snnn, hosvd_result=None)

    # ── Core interface for P-RHPG ────────────────────────────

    @property
    def vertices(self):
        """List of (A_i, B_i) tuples, length N_v."""
        verts, _ = self.snnn.get_vertex_systems()
        return verts

    @property
    def A_vertices(self):
        """(N_v, n, n) array of vertex A matrices."""
        return np.array([A for A, B in self.vertices])

    @property
    def B_vertices(self):
        """(N_v, n, m) array of vertex B matrices."""
        return np.array([B for A, B in self.vertices])

    def weights(self, p):
        """Evaluate α_i(p) at a scheduling parameter point.

        Parameters
        ----------
        p : (n_p,) array

        Returns
        -------
        alpha : (N_v,) array — non-negative, sums to 1
        """
        return self.snnn.evaluate_weights(p)

    def evaluate(self, p):
        """Evaluate [A(p) B(p)] at a scheduling parameter point.

        Parameters
        ----------
        p : (n_p,) array

        Returns
        -------
        S : (n, n+m) array — [A(p) | B(p)]
        """
        return self.snnn.evaluate_system(p)

    def A(self, p):
        """State matrix A(p) at a scheduling parameter."""
        S = self.evaluate(p)
        return S[:, : self.n]

    def B(self, p):
        """Input matrix B(p) at a scheduling parameter."""
        S = self.evaluate(p)
        return S[:, self.n :]

    # ── Closed-loop simulation ───────────────────────────────

    def simulate_polytopic(self, K_vertices, x0, p_trajectory, N=None):
        """Simulate the polytopic closed-loop system.

        x_{t+1} = A(p_t) x_t + B(p_t) u_t
        u_t = K(p_t) x_t = [Σ α_i(p_t) K_i] x_t

        Parameters
        ----------
        K_vertices : (N_v, m, n) array — vertex gain matrices
        x0 : (n,) array — initial state
        p_trajectory : (T, n_p) array — scheduling parameter trajectory
        N : int, optional — override trajectory length

        Returns
        -------
        x_traj : (T+1, n) array — state trajectory
        u_traj : (T, m) array — input trajectory
        cost : float — Σ (x'Qx + u'Ru) with Q=I, R=I
        """
        T = len(p_trajectory) if N is None else N
        x_traj = np.zeros((T + 1, self.n))
        u_traj = np.zeros((T, self.m))
        x_traj[0] = x0

        for t in range(T):
            p_t = p_trajectory[t]
            alpha = self.weights(p_t)

            # Blended gain
            K_t = np.zeros((self.m, self.n))
            for i, a in enumerate(alpha):
                K_t += a * K_vertices[i]

            # System at this parameter
            A_t = self.A(p_t)
            B_t = self.B(p_t)

            u_traj[t] = K_t @ x_traj[t]
            x_traj[t + 1] = A_t @ x_traj[t] + B_t @ u_traj[t]

        return x_traj, u_traj

    def simulate_nonlinear(self, K_vertices, x0, T, Q=None, R=None):
        """Simulate the nonlinear closed-loop (if system supports it).

        Uses system.nonlinear_step(x, u) and system.scheduling_param(x).

        Parameters
        ----------
        K_vertices : (N_v, m, n) array
        x0 : (n,) array
        T : int — number of steps
        Q, R : cost matrices (default: identity)

        Returns
        -------
        x_traj : (T+1, n) array
        u_traj : (T, m) array
        cost : float
        """
        if Q is None:
            Q = np.eye(self.n)
        if R is None:
            R = np.eye(self.m)

        x_traj = np.zeros((T + 1, self.n))
        u_traj = np.zeros((T, self.m))
        x_traj[0] = x0
        cost = 0.0

        for t in range(T):
            p_t = self.system.scheduling_param(x_traj[t])
            alpha = self.weights(p_t)

            K_t = np.zeros((self.m, self.n))
            for i, a in enumerate(alpha):
                K_t += a * K_vertices[i]

            u_traj[t] = K_t @ x_traj[t]
            x_traj[t + 1] = self.system.nonlinear_step(x_traj[t], u_traj[t])

            cost += x_traj[t] @ Q @ x_traj[t] + u_traj[t] @ R @ u_traj[t]

        cost += x_traj[T] @ Q @ x_traj[T]
        return x_traj, u_traj, cost

    # ── Diagnostics ──────────────────────────────────────────

    def gram_matrix(self, fine_grid=None):
        """Gram matrix Γ_{ij} = ∫ α_i(p) α_j(p) dμ(p).

        If fine_grid is provided, evaluates on it. Otherwise uses the
        vertex grid (exact for multilinear, approximate for others).

        Returns
        -------
        Gamma : (N_v, N_v) array
        kappa : float — condition number
        """
        if fine_grid is None:
            Gamma = self.snnn.gram_matrix()
        else:
            N = fine_grid.total_points
            n_p = fine_grid.n_p
            alpha = np.zeros((N, self.n_vertices))
            for i in range(N):
                mi = np.unravel_index(i, fine_grid.n_points)
                p = np.array([fine_grid.axes[k][mi[k]] for k in range(n_p)])
                alpha[i] = self.weights(p)
            mu = np.ones(N) / N
            Gamma = alpha.T @ np.diag(mu) @ alpha

        eigs = np.linalg.eigvalsh(Gamma)
        kappa = eigs[-1] / eigs[0] if eigs[0] > 1e-15 else np.inf
        return Gamma, kappa

    def reconstruction_error(self, test_grid=None):
        """Measure polytopic model accuracy vs true system.

        Parameters
        ----------
        test_grid : ParameterGrid, optional
            Fine grid for evaluation. Default: 2× vertex grid density.

        Returns
        -------
        max_err, rel_err : float
        """
        if test_grid is None:
            test_n = [max(2 * k, 10) for k in self.snnn.ranks]
            test_grid = ParameterGrid(self.system.param_bounds, n_points=test_n)

        return multilinear_error(self.system, self.vertex_grid, test_grid)

    def vertex_stability(self):
        """Check spectral radius of each vertex A_i.

        Returns
        -------
        radii : (N_v,) array — spectral radii
        all_stable : bool — True if all ρ(A_i) < 1
        """
        radii = []
        for A, B in self.vertices:
            eigs = np.abs(np.linalg.eigvals(A))
            radii.append(np.max(eigs))
        radii = np.array(radii)
        return radii, np.all(radii < 1.0)

    def summary(self):
        """Print model summary."""
        print(f"\nTPModel Summary:")
        print(f"  System: n={self.n}, m={self.m}, n_p={self.n_p}")
        print(f"  Grid: {list(self.snnn.ranks)} = {self.n_vertices} vertices")
        print(f"  Param bounds: {self.system.param_bounds}")

        # Gram
        _, kappa = self.gram_matrix()
        print(f"  Gram κ(Γ) = {kappa:.1f}")

        # Reconstruction
        me, re = self.reconstruction_error()
        print(f"  Reconstruction: rel_err = {re:.2e}")

        # Stability
        radii, stable = self.vertex_stability()
        print(f"  Vertex stability: ρ_max = {radii.max():.4f}, all open-loop stable = {stable}")

        # HOSVD ranks if available
        if self.hosvd_result is not None:
            print(f"  HOSVD ranks: {self.hosvd_result.ranks}")


# ── Quick test ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")

    print("=" * 60)
    print("TPModel — Tests")
    print("=" * 60)

    # --- Pendulum ---
    from systems.inverted_pendulum import InvertedPendulum

    pend = InvertedPendulum()

    print("\n--- Pendulum: from_system (recommended) ---")
    tp_p = TPModel.from_system(pend, n_sample=[40, 30], threshold=0.001)

    # Basic interface
    p_test = np.array([0.3, 1.0])
    alpha = tp_p.weights(p_test)
    assert np.abs(alpha.sum() - 1.0) < 1e-12
    assert np.all(alpha >= -1e-12)
    A_p = tp_p.A(p_test)
    B_p = tp_p.B(p_test)
    assert A_p.shape == (4, 4)
    assert B_p.shape == (4, 1)
    print(f"  weights({p_test}): sum={alpha.sum():.10f}, nneg=✓")
    print(f"  A shape={A_p.shape}, B shape={B_p.shape}")

    # Vertex access
    verts = tp_p.vertices
    print(f"  {len(verts)} vertices, A_i: {verts[0][0].shape}")
    A_all = tp_p.A_vertices
    B_all = tp_p.B_vertices
    print(f"  A_vertices: {A_all.shape}, B_vertices: {B_all.shape}")

    # --- Pendulum: from_grid (explicit) ---
    print("\n--- Pendulum: from_grid [6, 4] ---")
    tp_p2 = TPModel.from_grid(pend, [6, 4])
    tp_p2.summary()

    # --- Wing ---
    from systems.aeroelastic_wing import AeroelasticWing

    wing = AeroelasticWing()

    print("\n--- Wing: from_system ---")
    tp_w = TPModel.from_system(wing, n_sample=[20, 5, 5], threshold=0.001)

    # Gram conditioning
    _, kappa = tp_w.gram_matrix()
    print(f"  Gram κ = {kappa:.1f}")

    fine = ParameterGrid(wing.param_bounds, n_points=[40, 10, 10])
    _, kappa_fine = tp_w.gram_matrix(fine_grid=fine)
    print(f"  Gram κ (fine grid) = {kappa_fine:.1f}")

    # --- Simulate with random stabilizing gains ---
    print("\n--- Polytopic simulation test ---")
    Nv = tp_p.n_vertices
    # Use zero gain (just to test mechanics)
    K_zero = np.zeros((Nv, 1, 4))
    x0 = np.array([0.0, 0.0, 0.3, 0.0])
    # Generate parameter trajectory from state
    p_traj = np.array([pend.scheduling_param(x0)] * 50)
    x_traj, u_traj = tp_p.simulate_polytopic(K_zero, x0, p_traj)
    print(f"  x_traj shape: {x_traj.shape}, u_traj shape: {u_traj.shape}")
    print(f"  x_traj[0] = {x_traj[0]}")
    print(f"  ||x_traj[-1]|| = {np.linalg.norm(x_traj[-1]):.4f}")

    # --- Nonlinear simulation (pendulum has scheduling_param) ---
    print("\n--- Nonlinear simulation test ---")
    x_nl, u_nl, cost_nl = tp_p.simulate_nonlinear(K_zero, x0, T=50)
    print(f"  x_nl shape: {x_nl.shape}")
    print(f"  cost (zero gain) = {cost_nl:.2f}")

    print("\nAll tests passed.")
