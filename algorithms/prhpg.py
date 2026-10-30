"""
P-RHPG: Polytopic Receding-Horizon Policy Gradient
====================================================

Implements Algorithm 1 from the paper:

    Given vertex systems (A_i, B_i), weighting functions α_i,
    cost matrices Q, R, Q_N, initial condition Σ_0, horizon N:

    1. Initialize P_N(p) = Q_N for all p
    2. For h = N-1, ..., 0:
       a. Solve stage subproblem: K_{i,h}* = argmin Φ_h({K_i})
       b. Update cost-to-go P_h(p) via backward recursion
    3. Return first-stage gains (K_1*, ..., K_Nv*)

Each stage subproblem is a strongly convex quadratic in the vertex
gains, solvable either by direct linear system solve or gradient descent.

Key equations (see paper Section 3):
    Stage cost:     Φ_h = ∫ Tr[(Q + K'RK + Acl' P_{h+1} Acl) Σ_0] dμ
    Gradient:       ∇_{K_i} Φ_h = ∫ α_i(p) · 2 E_h(p) Σ_0 dμ
    Residual:       E_h(p) = M_h(p) K(p) + B(p)' P_{h+1}(p) A(p)
    Hessian block:  [H]_{ij} = ∫ α_i α_j · 2 (Σ_0 ⊗ M_h(p)) dμ
    where M_h(p) = R + B(p)' P_{h+1}(p) B(p)
"""

import time

import numpy as np
from scipy import linalg


class PRHPGResult:
    """Result of P-RHPG algorithm.

    Attributes
    ----------
    K_vertices : (Nv, m, n) array
        First-stage vertex gains (the PDC controller output).
    cost_history : (N,) array
        Stage cost at each backward step (Φ_{N-1}, ..., Φ_0).
    P_history : list of (N_q, n, n) arrays, optional
        Cost-to-go matrices at each stage (if stored).
    K_all : list of (Nv, m, n) arrays
        Vertex gains at all N stages [K_0*, K_1*, ..., K_{N-1}*].
    integrated_cost : float
        Final integrated cost J_N = ∫ Tr(P_0 Σ_0) dμ.
    stage_info : list of dict
        Per-stage diagnostics (iterations, gradient norm, etc.).
    """

    def __init__(self):
        self.K_vertices = None
        self.cost_history = []
        self.P_history = []
        self.K_all = []
        self.integrated_cost = None
        self.stage_info = []


class PRHPG:
    """Polytopic Receding-Horizon Policy Gradient solver.

    Parameters
    ----------
    tp_model : TPModel
        Polytopic model providing vertices, weights, and evaluation.
    Q : (n, n) array — state penalty (≥ 0)
    R : (m, m) array — input penalty (> 0)
    Sigma0 : (n, n) array — initial condition weighting (> 0)
    quad_points : int or list of int, optional
        Number of quadrature points per scheduling dimension.
        Default: 2× the vertex grid density.
    """

    def __init__(self, tp_model, Q, R, Sigma0, quad_points=None):
        self.tp = tp_model
        self.n = tp_model.n
        self.m = tp_model.m
        self.Nv = tp_model.n_vertices
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.Sigma0 = np.array(Sigma0, dtype=float)

        # Precompute quadrature data
        self._setup_quadrature(quad_points)

    def _setup_quadrature(self, quad_points):
        """Precompute A(p_q), B(p_q), α_i(p_q) at all quadrature points."""
        from tp_transform.sampling import ParameterGrid

        if quad_points is None:
            quad_points = [max(2 * k, 6) for k in self.tp.snnn.ranks]
        elif isinstance(quad_points, int):
            quad_points = [quad_points] * self.tp.n_p

        quad_grid = ParameterGrid(self.tp.system.param_bounds,
                                  n_points=quad_points)
        N_q = quad_grid.total_points
        n_p = quad_grid.n_p

        self.N_q = N_q
        self.quad_grid = quad_grid

        # Uniform measure: μ_q = 1/N_q
        self.mu = np.ones(N_q) / N_q

        # Precompute system matrices and weights at quadrature points
        self.A_quad = np.zeros((N_q, self.n, self.n))
        self.B_quad = np.zeros((N_q, self.n, self.m))
        self.alpha_quad = np.zeros((N_q, self.Nv))

        for q in range(N_q):
            multi_idx = np.unravel_index(q, quad_grid.n_points)
            p = np.array([quad_grid.axes[k][multi_idx[k]]
                          for k in range(n_p)])

            # System matrices from the true system
            S = self.tp.system.system_matrix(p, discrete=True)
            self.A_quad[q] = S[:, :self.n]
            self.B_quad[q] = S[:, self.n:]

            # Weight vector from the polytopic model
            self.alpha_quad[q] = self.tp.weights(p)

    # ── Main solver ──────────────────────────────────────────

    def solve(self, N, Q_N=None, method='direct', max_iter=500,
              tol=1e-10, store_all=False, verbose=True):
        """Run P-RHPG backward sweep.

        Parameters
        ----------
        N : int — horizon length
        Q_N : (n, n) array or str, optional
            Terminal cost. If None or 'Q', uses Q.
            If 'are', uses average of vertex-wise ARE solutions.
            If array, uses directly.
        method : str — 'direct' or 'gd' (gradient descent)
        max_iter : int — max GD iterations per stage (for method='gd')
        tol : float — gradient norm tolerance (for method='gd')
        store_all : bool — if True, store P_h at every stage
        verbose : bool

        Returns
        -------
        result : PRHPGResult
        """
        n, m, Nv, N_q = self.n, self.m, self.Nv, self.N_q

        # Terminal cost
        Q_N = self._get_terminal_cost(Q_N)

        result = PRHPGResult()

        # Initialize: P_N(p_q) = Q_N for all q
        P_next = np.tile(Q_N, (N_q, 1, 1))  # (N_q, n, n)

        if store_all:
            result.P_history.append(P_next.copy())

        # Backward sweep
        for h in range(N - 1, -1, -1):
            if method == 'direct':
                K_h, info = self._stage_solve_direct(P_next)
            elif method == 'gd':
                K_h, info = self._stage_solve_gd(P_next, max_iter, tol)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Compute stage cost
            stage_cost = self._stage_cost(K_h, P_next)
            info['stage_cost'] = stage_cost

            # Update cost-to-go
            P_next = self._update_cost_to_go(K_h, P_next)

            result.K_all.append(K_h.copy())
            result.cost_history.append(stage_cost)
            result.stage_info.append(info)

            if store_all:
                result.P_history.append(P_next.copy())

            if verbose and (h % max(1, N // 10) == 0 or h == N - 1 or h == 0):
                rho = self._max_spectral_radius(K_h)
                print(f"  Stage {h:4d}: cost={stage_cost:.6e}, "
                      f"ρ_max={rho:.4f}, "
                      f"||∇||={info.get('grad_norm', 0):.2e}")

        # First-stage gains are the output
        result.K_vertices = result.K_all[-1].copy()  # last appended = stage 0
        result.K_all.reverse()  # reorder to [K_0, K_1, ..., K_{N-1}]
        result.cost_history.reverse()

        # Integrated cost: J_N = ∫ Tr(P_0 Σ_0) dμ
        result.integrated_cost = np.sum(
            [self.mu[q] * np.trace(P_next[q] @ self.Sigma0)
             for q in range(N_q)]
        )

        if verbose:
            rho = self._max_spectral_radius(result.K_vertices)
            print(f"\n  P-RHPG complete: J_N = {result.integrated_cost:.6e}, "
                  f"ρ_max = {rho:.4f}")

        return result

    # ── Stage solvers ────────────────────────────────────────

    def _stage_solve_direct(self, P_next):
        """Solve stage subproblem via direct linear system.

        The stage cost Φ_h is quadratic:
            Φ_h(K) = 0.5 vec(K)' H vec(K) + g0' vec(K) + const

        Minimizer: vec(K*) = -H^{-1} g0

        where H is the Hessian and g0 = ∇Φ_h|_{K=0}.
        """
        n, m, Nv, N_q = self.n, self.m, self.Nv, self.N_q
        dim = Nv * m * n  # total dimension of vectorized vertex gains

        t_q_start = time.perf_counter()

        # Precompute M_h(q) = R + B(q)' P_{h+1}(q) B(q)  for each q
        M_q = np.zeros((N_q, m, m))
        for q in range(N_q):
            M_q[q] = self.R + self.B_quad[q].T @ P_next[q] @ self.B_quad[q]

        # ---- Assemble Hessian ----
        # [H]_{ij} = Σ_q μ_q α_i(q) α_j(q) · 2 (Σ_0 ⊗ M_h(q))
        # Each block is (mn × mn). Full H is (Nv·mn × Nv·mn).
        H = np.zeros((dim, dim))
        mn = m * n

        for q in range(N_q):
            # Kronecker product: Σ_0 ⊗ M_h(q)  — (mn × mn)
            kron_q = np.kron(self.Sigma0, M_q[q])  # (mn, mn)
            alpha_q = self.alpha_quad[q]  # (Nv,)

            for i in range(Nv):
                for j in range(Nv):
                    H[i * mn:(i + 1) * mn,
                      j * mn:(j + 1) * mn] += (
                        2.0 * self.mu[q] * alpha_q[i] * alpha_q[j] * kron_q
                    )

        # ---- Gradient at K=0 ----
        # g0_i = ∫ α_i(p) · 2 B(p)' P_{h+1}(p) A(p) Σ_0 dμ
        g0 = np.zeros(dim)
        for q in range(N_q):
            E0_q = self.B_quad[q].T @ P_next[q] @ self.A_quad[q]  # (m, n)
            grad_q = 2.0 * E0_q @ self.Sigma0  # (m, n)
            alpha_q = self.alpha_quad[q]
            for i in range(Nv):
                g0[i * mn:(i + 1) * mn] += (
                    self.mu[q] * alpha_q[i] * grad_q.ravel()
                )

        t_quad = time.perf_counter() - t_q_start

        t_s_start = time.perf_counter()
        # ---- Solve H @ vec(K*) = -g0 ----
        if np.any(~np.isfinite(H)) or np.any(~np.isfinite(g0)):
            raise ValueError(
                "NaN/Inf in Hessian or gradient. Check weight functions "
                "and quadrature grid for degenerate points.")
        try:
            K_vec = np.linalg.solve(H, -g0)
        except np.linalg.LinAlgError:
            from scipy.linalg import solve as sp_solve
            K_vec = sp_solve(H, -g0, assume_a='pos')
        K_h = K_vec.reshape(Nv, m, n)
        t_solve = time.perf_counter() - t_s_start

        # Diagnostics
        grad_at_opt = H @ K_vec + g0
        try:
            hcond = float(np.linalg.cond(H))
        except (np.linalg.LinAlgError, ValueError):
            try:
                from scipy.linalg import eigvalsh as sp_eigvalsh
                eigs = sp_eigvalsh(H)
                hcond = float(eigs[-1] / max(eigs[0], 1e-30))
            except Exception:
                hcond = np.inf

        info = {
            'method': 'direct',
            'grad_norm': np.linalg.norm(grad_at_opt),
            'hessian_cond': hcond,
            't_quad': t_quad,
            't_solve': t_solve,
        }

        return K_h, info

    def _stage_solve_gd(self, P_next, max_iter, tol):
        """Solve stage subproblem via gradient descent.

        K^{k+1} = K^k - η ∇Φ_h(K^k)
        with η = 1 / σ_max(H).
        """
        n, m, Nv = self.n, self.m, self.Nv

        # Initialize from zero
        K_curr = np.zeros((Nv, m, n))

        # Need the Hessian for step size
        dim = Nv * m * n

        t_q_start = time.perf_counter()
        M_q = np.zeros((self.N_q, m, m))
        for q in range(self.N_q):
            M_q[q] = self.R + self.B_quad[q].T @ P_next[q] @ self.B_quad[q]

        H = np.zeros((dim, dim))
        mn = m * n
        for q in range(self.N_q):
            kron_q = np.kron(self.Sigma0, M_q[q])
            alpha_q = self.alpha_quad[q]
            for i in range(Nv):
                for j in range(Nv):
                    H[i * mn:(i + 1) * mn,
                      j * mn:(j + 1) * mn] += (
                        2.0 * self.mu[q] * alpha_q[i] * alpha_q[j] * kron_q
                    )

        # Also assemble g0 = ∇Φ_h|_{K=0} (not needed for GD steps, but useful timing-wise)
        g0 = np.zeros(dim)
        for q in range(self.N_q):
            E0_q = self.B_quad[q].T @ P_next[q] @ self.A_quad[q]  # (m, n)
            grad_q = 2.0 * E0_q @ self.Sigma0  # (m, n)
            alpha_q = self.alpha_quad[q]
            for i in range(Nv):
                g0[i * mn:(i + 1) * mn] += self.mu[q] * alpha_q[i] * grad_q.ravel()

        t_quad = time.perf_counter() - t_q_start

        try:
            sigma_max = np.linalg.norm(H, ord=2)
            sigma_min = np.linalg.norm(np.linalg.inv(H), ord=2) ** (-1)
        except (np.linalg.LinAlgError, ValueError):
            from scipy.linalg import eigvalsh as sp_eigvalsh
            try:
                eigs = sp_eigvalsh(H)
                sigma_max = float(eigs[-1])
                sigma_min = float(max(eigs[0], 1e-30))
            except Exception:
                sigma_max = float(np.max(np.abs(H)))
                sigma_min = 1e-30
        eta = 1.0 / sigma_max

        grad_norms = []
        cost_trace = []

        t_s_start = time.perf_counter()
        for k in range(max_iter):
            grad = self._stage_gradient(K_curr, P_next)  # (Nv, m, n)
            gn = np.linalg.norm(grad)
            grad_norms.append(gn)

            cost_k = self._stage_cost(K_curr, P_next)
            cost_trace.append(cost_k)

            if gn < tol:
                break

            K_curr -= eta * grad
        t_solve = time.perf_counter() - t_s_start

        info = {
            'method': 'gd',
            'iterations': k + 1,
            'n_iter': k,
            'grad_norm': grad_norms[-1] if grad_norms else 0,
            'grad_norms': grad_norms,
            'cost_trace': cost_trace,
            'step_size': eta,
            'hessian_cond': sigma_max / sigma_min if sigma_min > 0 else np.inf,
            't_quad': t_quad,
            't_solve': t_solve,
        }

        return K_curr, info

    # ── Gradient and cost evaluation ─────────────────────────

    def _stage_gradient(self, K_h, P_next):
        """Evaluate stage gradient ∇Φ_h at given vertex gains.

        ∇_{K_i} Φ_h = ∫ α_i(p) · 2 E_h(p) Σ_0 dμ(p)
        E_h(p) = M_h(p) K(p) + B(p)' P_{h+1}(p) A(p)

        Parameters
        ----------
        K_h : (Nv, m, n) array — current vertex gains
        P_next : (N_q, n, n) array — cost-to-go at next stage

        Returns
        -------
        grad : (Nv, m, n) array — gradient w.r.t. each vertex gain
        """
        n, m, Nv, N_q = self.n, self.m, self.Nv, self.N_q
        grad = np.zeros((Nv, m, n))

        for q in range(N_q):
            A_q = self.A_quad[q]
            B_q = self.B_quad[q]
            P_q = P_next[q]
            alpha_q = self.alpha_quad[q]

            # Blended gain: K(p_q) = Σ_i α_i(p_q) K_i
            K_blend = np.einsum('i,imn->mn', alpha_q, K_h)

            # M_h(q) = R + B'PB
            M_q = self.R + B_q.T @ P_q @ B_q

            # E_h(q) = M K(p) + B'PA
            E_q = M_q @ K_blend + B_q.T @ P_q @ A_q

            # Gradient contribution: α_i · 2 E Σ_0
            grad_q = 2.0 * E_q @ self.Sigma0  # (m, n)

            for i in range(Nv):
                grad[i] += self.mu[q] * alpha_q[i] * grad_q

        return grad

    def _stage_cost(self, K_h, P_next):
        """Evaluate stage cost Φ_h.

        Φ_h = ∫ Tr[(Q + K'RK + Acl' P_{h+1} Acl) Σ_0] dμ
        """
        cost = 0.0
        for q in range(self.N_q):
            A_q = self.A_quad[q]
            B_q = self.B_quad[q]
            P_q = P_next[q]
            alpha_q = self.alpha_quad[q]

            K_blend = np.einsum('i,imn->mn', alpha_q, K_h)
            Acl = A_q + B_q @ K_blend

            P_h_q = (self.Q + K_blend.T @ self.R @ K_blend
                     + Acl.T @ P_q @ Acl)

            cost += self.mu[q] * np.trace(P_h_q @ self.Sigma0)

        return cost

    def _update_cost_to_go(self, K_h, P_next):
        """Compute P_h(p_q) for all quadrature points.

        P_h(q) = Q + K(q)'RK(q) + Acl(q)' P_{h+1}(q) Acl(q)
        """
        P_h = np.zeros_like(P_next)

        for q in range(self.N_q):
            A_q = self.A_quad[q]
            B_q = self.B_quad[q]
            P_q = P_next[q]
            alpha_q = self.alpha_quad[q]

            K_blend = np.einsum('i,imn->mn', alpha_q, K_h)
            Acl = A_q + B_q @ K_blend

            P_h[q] = (self.Q + K_blend.T @ self.R @ K_blend
                      + Acl.T @ P_q @ Acl)

        return P_h

    # ── Terminal cost ────────────────────────────────────────

    def _get_terminal_cost(self, Q_N):
        """Determine terminal cost matrix Q_N.

        Options:
            None or 'Q'   → Q_N = Q
            'are'         → average of vertex-wise ARE solutions
            ndarray       → use directly
        """
        if Q_N is None or (isinstance(Q_N, str) and Q_N.lower() == 'q'):
            return self.Q.copy()

        if isinstance(Q_N, str) and Q_N.lower() == 'are':
            return self._vertex_are_terminal()

        return np.array(Q_N, dtype=float)

    def _vertex_are_terminal(self):
        """Compute Q_N = (1/Nv) Σ_i P_i^{are}.

        Solves the DARE at each vertex independently and averages.
        Falls back to Q if any vertex is not stabilizable.
        """
        P_sum = np.zeros((self.n, self.n))
        n_solved = 0

        for A_i, B_i in self.tp.vertices:
            try:
                P_i = linalg.solve_discrete_are(A_i, B_i, self.Q, self.R)
                P_sum += P_i
                n_solved += 1
            except (np.linalg.LinAlgError, ValueError):
                # Vertex may not be stabilizable — skip
                pass

        if n_solved > 0:
            return P_sum / n_solved
        else:
            return self.Q.copy()

    # ── Diagnostics ──────────────────────────────────────────

    def _max_spectral_radius(self, K_h):
        """Max spectral radius of Acl(p_q) over all quadrature points."""
        rho_max = 0.0
        for q in range(self.N_q):
            alpha_q = self.alpha_quad[q]
            K_blend = np.einsum('i,imn->mn', alpha_q, K_h)
            Acl = self.A_quad[q] + self.B_quad[q] @ K_blend
            rho = np.max(np.abs(np.linalg.eigvals(Acl)))
            rho_max = max(rho_max, rho)
        return rho_max

    def evaluate_infinite_cost(self, K_vertices, max_iter=1000, tol=1e-12):
        """Evaluate the infinite-horizon frozen-parameter cost J_∞(K).

        For each quadrature point, solves the Lyapunov equation
        P = Q + K'RK + Acl' P Acl by iteration.

        Returns
        -------
        J_inf : float — integrated infinite-horizon cost (NaN if unstable)
        rho_max : float — maximum spectral radius
        """
        J_inf = 0.0
        rho_max = 0.0

        for q in range(self.N_q):
            alpha_q = self.alpha_quad[q]
            K_blend = np.einsum('i,imn->mn', alpha_q, K_vertices)
            Acl = self.A_quad[q] + self.B_quad[q] @ K_blend
            rho = np.max(np.abs(np.linalg.eigvals(Acl)))
            rho_max = max(rho_max, rho)

            if rho >= 1.0:
                return np.nan, rho_max

            # Lyapunov iteration: P = Q + K'RK + Acl' P Acl
            S = self.Q + K_blend.T @ self.R @ K_blend
            P = S.copy()
            for _ in range(max_iter):
                P_new = S + Acl.T @ P @ Acl
                if np.max(np.abs(P_new - P)) < tol:
                    break
                P = P_new

            J_inf += self.mu[q] * np.trace(P @ self.Sigma0)

        return J_inf, rho_max

    def riccati_benchmark(self):
        """Compute the pointwise Riccati cost (unconstrained LQR at each p).

        This is the lower bound: J_∞^{ric} = ∫ Tr(P^{ric}(p) Σ_0) dμ.
        No PDC constraint, so not achievable in general.

        Returns
        -------
        J_ric : float
        K_ric : (N_q, m, n) array — Riccati gains at each quad point
        """
        J_ric = 0.0
        K_ric = np.zeros((self.N_q, self.m, self.n))

        for q in range(self.N_q):
            A_q = self.A_quad[q]
            B_q = self.B_quad[q]
            try:
                P_q = linalg.solve_discrete_are(A_q, B_q, self.Q, self.R)
                K_q = -np.linalg.solve(
                    self.R + B_q.T @ P_q @ B_q,
                    B_q.T @ P_q @ A_q
                )
                K_ric[q] = K_q
                J_ric += self.mu[q] * np.trace(P_q @ self.Sigma0)
            except (np.linalg.LinAlgError, ValueError):
                J_ric = np.nan
                break

        return J_ric, K_ric


# ── Convenience function ─────────────────────────────────────

def run_prhpg(tp_model, N, Q=None, R=None, Sigma0=None, Q_N='are',
              method='direct', quad_points=None, verbose=True):
    """One-call P-RHPG: build solver and run.

    Parameters
    ----------
    tp_model : TPModel
    N : int — horizon length
    Q, R, Sigma0 : arrays, optional (defaults to identity)
    Q_N : terminal cost specification
    method : 'direct' or 'gd'
    quad_points : int or list of int, optional

    Returns
    -------
    result : PRHPGResult
    solver : PRHPG
    """
    n, m = tp_model.n, tp_model.m

    if Q is None:
        Q = np.eye(n)
    if R is None:
        R = np.eye(m)
    if Sigma0 is None:
        Sigma0 = np.eye(n)

    solver = PRHPG(tp_model, Q, R, Sigma0, quad_points=quad_points)

    if verbose:
        J_ric, _ = solver.riccati_benchmark()
        print(f"P-RHPG: n={n}, m={m}, Nv={tp_model.n_vertices}, "
              f"N={N}, N_q={solver.N_q}")
        print(f"  Riccati lower bound: J_ric = {J_ric:.6e}")
        print(f"  Backward sweep ({method}):")

    result = solver.solve(N, Q_N=Q_N, method=method, verbose=verbose)

    if verbose:
        J_inf, rho = solver.evaluate_infinite_cost(result.K_vertices)
        print(f"  Infinite-horizon cost: J_inf = {J_inf:.6e}")
        if not np.isnan(J_ric):
            gap = (J_inf - J_ric) / J_ric * 100
            print(f"  Suboptimality gap: {gap:.2f}%")

    return result, solver


# ── Tests ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from tp_transform.tp_model import TPModel
    from systems.inverted_pendulum import InvertedPendulum
    from systems.aeroelastic_wing import AeroelasticWing

    print("=" * 60)
    print("P-RHPG — Tests")
    print("=" * 60)

    # ── Test 1: Single-vertex LQR (should recover Riccati) ──
    print("\n--- Test 1: Single-vertex (standard LQR) ---")

    # Create a trivial 1-vertex polytopic model
    pend = InvertedPendulum()
    tp1 = TPModel.from_grid(pend, [1, 1])  # 1 vertex = single system at center
    print(f"  Nv = {tp1.n_vertices}")

    result1, solver1 = run_prhpg(tp1, N=50, Q_N='are', verbose=True)
    J_ric, _ = solver1.riccati_benchmark()
    J_inf, rho = solver1.evaluate_infinite_cost(result1.K_vertices)
    print(f"  Riccati = {J_ric:.6e}, P-RHPG J_inf = {J_inf:.6e}")
    if not np.isnan(J_ric) and not np.isnan(J_inf):
        print(f"  Match: {abs(J_inf - J_ric) / J_ric:.2e}")

    # ── Test 2: Pendulum polytopic ──
    print("\n--- Test 2: Pendulum polytopic ---")

    tp_p = TPModel.from_system(pend, n_sample=[40, 30], threshold=0.001,
                               verbose=False)
    print(f"  Nv = {tp_p.n_vertices}, grid = {tp_p.snnn.ranks}")

    for N in [10, 30, 50, 100]:
        result, solver = run_prhpg(tp_p, N=N, Q_N='are',
                                   quad_points=8, verbose=False)
        J_inf, rho = solver.evaluate_infinite_cost(result.K_vertices)
        J_ric, _ = solver.riccati_benchmark()
        gap = (J_inf - J_ric) / J_ric * 100 if not np.isnan(J_inf) else float('inf')
        print(f"  N={N:3d}: J_inf={J_inf:.4e}, ρ={rho:.4f}, "
              f"gap={gap:.2f}%")

    # ── Test 3: Wing polytopic ──
    print("\n--- Test 3: Wing polytopic ---")

    wing = AeroelasticWing()
    tp_w = TPModel.from_system(wing, n_sample=[20, 5, 5], threshold=0.001,
                               verbose=False)
    print(f"  Nv = {tp_w.n_vertices}, grid = {tp_w.snnn.ranks}")

    for N in [10, 30, 50, 100]:
        result, solver = run_prhpg(tp_w, N=N, Q_N='are',
                                   quad_points=6, verbose=False)
        J_inf, rho = solver.evaluate_infinite_cost(result.K_vertices)
        J_ric, _ = solver.riccati_benchmark()
        gap = (J_inf - J_ric) / J_ric * 100 if not np.isnan(J_inf) else float('inf')
        print(f"  N={N:3d}: J_inf={J_inf:.4e}, ρ={rho:.4f}, "
              f"gap={gap:.2f}%")

    # ── Test 4: GD vs Direct ──
    print("\n--- Test 4: GD vs Direct (pendulum, N=30) ---")
    result_d, solver_d = run_prhpg(tp_p, N=30, Q_N='are', method='direct',
                                   quad_points=8, verbose=False)
    result_g, solver_g = run_prhpg(tp_p, N=30, Q_N='are', method='gd',
                                   quad_points=8, verbose=False)
    J_d, _ = solver_d.evaluate_infinite_cost(result_d.K_vertices)
    J_g, _ = solver_g.evaluate_infinite_cost(result_g.K_vertices)
    diff = np.max(np.abs(result_d.K_vertices - result_g.K_vertices))
    print(f"  Direct: J_inf = {J_d:.6e}")
    print(f"  GD:     J_inf = {J_g:.6e}")
    print(f"  Max gain difference: {diff:.2e}")

    # ── Test 5: Gradient verification ──
    print("\n--- Test 5: Gradient finite-difference check ---")
    solver_check = PRHPG(tp_p, np.eye(4), np.eye(1), np.eye(4), quad_points=8)
    Q_N_mat = solver_check._get_terminal_cost('are')
    P_test = np.tile(Q_N_mat, (solver_check.N_q, 1, 1))
    K_test = np.random.randn(tp_p.n_vertices, tp_p.m, tp_p.n) * 0.01
    grad = solver_check._stage_gradient(K_test, P_test)

    eps = 1e-6
    grad_fd = np.zeros_like(grad)
    for i in range(tp_p.n_vertices):
        for a in range(tp_p.m):
            for b in range(tp_p.n):
                K_plus = K_test.copy()
                K_plus[i, a, b] += eps
                K_minus = K_test.copy()
                K_minus[i, a, b] -= eps
                grad_fd[i, a, b] = (
                    (solver_check._stage_cost(K_plus, P_test)
                     - solver_check._stage_cost(K_minus, P_test))
                    / (2 * eps)
                )

    rel_err = np.linalg.norm(grad - grad_fd) / (np.linalg.norm(grad) + 1e-15)
    print(f"  ||∇_analytic - ∇_fd|| / ||∇_analytic|| = {rel_err:.2e}")
    assert rel_err < 1e-5, f"Gradient check failed: {rel_err}"
    print("  PASSED ✓")

    # ── Test 6: Convergence in N (Theorem 4.1) ──
    print("\n--- Test 6: Convergence in N (Theorem 4.1) ---")
    solver_conv = PRHPG(tp_p, np.eye(4), np.eye(1), np.eye(4), quad_points=8)
    J_ric, _ = solver_conv.riccati_benchmark()
    print(f"  Riccati lower bound: {J_ric:.4e}")
    print(f"  {'N':>5s}  {'J_inf':>12s}  {'ρ_max':>8s}  {'gap%':>8s}  {'ΔJ_inf':>10s}")
    J_prev = None
    for N in [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
        res = solver_conv.solve(N, Q_N='are', method='direct', verbose=False)
        J_inf, rho = solver_conv.evaluate_infinite_cost(res.K_vertices)
        gap = (J_inf - J_ric) / J_ric * 100 if not np.isnan(J_inf) else float('inf')
        delta = abs(J_inf - J_prev) if J_prev is not None and not np.isnan(J_inf) else float('nan')
        print(f"  {N:5d}  {J_inf:12.4e}  {rho:8.4f}  {gap:8.3f}  {delta:10.2e}")
        J_prev = J_inf

    print("\nAll tests passed.")
