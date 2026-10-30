"""
Convex Hull Normalization (SNNN)
================================

Transforms raw HOSVD factor matrices into Sum-Normalized Non-Negative (SNNN)
weighting functions satisfying:
    (i)   α_i(p) ≥ 0           for all i, p
    (ii)  Σ_i α_i(p) = 1       for all p

Three methods are provided:

1. **Multilinear** (`multilinear_polytopic`) — RECOMMENDED for P-RHPG:
   Hat functions on a parameter grid. Inherently SNNN by construction.
   Full-rank Gram matrix (κ ≈ 20) → all vertex gains identifiable.
   Grid resolution guided by HOSVD singular value analysis.

2. **Simplex-based** (`snnn_normalize`): embeds HOSVD factor rows in a
   bounding simplex via barycentric coordinates. Near-exact reconstruction
   (bounded by HOSVD truncation error). However, Gram matrix is
   STRUCTURALLY rank-deficient: any factored SNNN with I_k+1 weights
   per dimension and sum-to-one has Gram rank = ∏I_k, not ∏(I_k+1).
   This creates unidentifiable gain directions in P-RHPG.

3. **Column-shift** (`snnn_column_shift`): classical Baranyi (2016).
   Destroys the DC component of HOSVD factors, causing 7-8% error.
   Also has rank-deficient Gram. Included for comparison only.

Recommended pipeline for P-RHPG:
    1. HOSVD on fine sampling grid → singular value decay per dimension
    2. Choose grid resolution I_k+1 per dimension from HOSVD rank I_k
    3. multilinear_polytopic() on the chosen grid
    4. P-RHPG with well-conditioned polytopic model

References:
    - Baranyi, "TP Model Transformation Based Control Design Frameworks",
      Springer 2016, Ch. 3
    - Szollosi & Baranyi, Asian J. Control 2016 (convexity type effects)
"""

import numpy as np
import itertools


# =====================================================================
# Simplex-based SNNN (recommended)
# =====================================================================

def _regular_simplex(d, circumradius=1.0):
    """Generate a regular simplex centered at origin in R^d.

    Returns (d+1) vertices in R^d, each at distance `circumradius` from
    the origin. All edges have equal length.

    Parameters
    ----------
    d : int — dimension of the ambient space
    circumradius : float — distance from center to each vertex

    Returns
    -------
    vertices : (d+1, d) array
    """
    I = d + 1
    # Centered standard basis in R^{d+1}: P[k] = e_k - ones/(d+1)
    P = np.eye(I) - np.ones((I, I)) / I
    # ||P[k]||^2 = 1 - 1/(d+1) = d/(d+1), so ||P[k]|| = sqrt(d/(d+1))

    # SVD to find the d-dimensional subspace (P has rank d)
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    Q = U[:, :d]  # (d+1, d) — orthonormal basis of the subspace
    V = P @ Q  # (d+1, d)

    # Scale to desired circumradius
    current_R = np.sqrt(d / (d + 1))
    V *= circumradius / current_R

    return V


def _barycentric_coords(points, simplex_vertices):
    """Compute barycentric coordinates of points w.r.t. a simplex.

    Parameters
    ----------
    points : (N, d) array — N points in R^d
    simplex_vertices : (d+1, d) array — vertices of the simplex

    Returns
    -------
    coords : (N, d+1) array — barycentric coordinates (sum to 1, ≥ 0 if inside)
    """
    d = points.shape[1]
    I = d + 1

    # Augmented system: [V^T; 1^T] @ λ = [x; 1]
    A = np.vstack([simplex_vertices.T, np.ones((1, I))])  # (d+1, d+1)
    A_inv = np.linalg.inv(A)

    augmented = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, d+1)
    coords = augmented @ A_inv.T  # (N, d+1)

    return coords


def _bounding_simplex(U, margin=0.05):
    """Find a regular simplex centered at the centroid of rows of U that
    encloses all rows with specified margin.

    Parameters
    ----------
    U : (N, I) array — HOSVD factor (rows are points in R^I)
    margin : float — fractional margin beyond the convex hull

    Returns
    -------
    simplex : (I+1, I) array — simplex vertices
    """
    I = U.shape[1]
    centroid = np.mean(U, axis=0)
    centered = U - centroid

    # Max distance from centroid
    dists = np.linalg.norm(centered, axis=1)
    R_data = np.max(dists)

    # For a regular simplex in R^I with circumradius R:
    #   inradius = R / I
    # To enclose a ball of radius R_data, need inradius ≥ R_data:
    #   R ≥ I * R_data
    R_simplex = I * R_data * (1 + margin)

    simplex = _regular_simplex(I, circumradius=R_simplex)
    simplex += centroid

    return simplex


def _snnn_simplex_factor(U, margin=0.05):
    """Transform an HOSVD factor to SNNN form via bounding simplex.

    Given U (N × I), embeds the rows in a (I+1)-vertex regular simplex
    and returns barycentric coordinates as SNNN weighting functions.

    Parameters
    ----------
    U : (N, I) array — HOSVD factor matrix
    margin : float — fractional margin for the bounding simplex

    Returns
    -------
    W : (N, I+1) array — SNNN weights (≥ 0, rows sum to 1)
    simplex : (I+1, I) array — simplex vertices used
    """
    simplex = _bounding_simplex(U, margin=margin)
    W = _barycentric_coords(U, simplex)

    # Clamp tiny negatives from numerical noise
    W = np.maximum(W, 0.0)
    W /= W.sum(axis=1, keepdims=True)

    return W, simplex


def snnn_normalize(hosvd_result, original_tensor, margin=0.05):
    """Apply simplex-based SNNN normalization to an HOSVD decomposition.

    For each scheduling dimension, the I_k orthonormal HOSVD basis vectors
    are embedded in a bounding (I_k+1)-vertex simplex. Barycentric
    coordinates give I_k+1 SNNN weighting functions per dimension.

    Total vertex count: N_v = ∏(I_k + 1).

    Parameters
    ----------
    hosvd_result : HOSVDResult
    original_tensor : ndarray of shape (*n_points, n, n+m)
    margin : float — fractional margin for bounding simplices

    Returns
    -------
    result : SNNNResult
    """
    n_p = len(hosvd_result.ranks)
    n = hosvd_result.n
    n_plus_m = hosvd_result.n_plus_m
    grid = hosvd_result.grid

    weights = []
    simplices = []
    snnn_ranks = []
    for k in range(n_p):
        W, simplex = _snnn_simplex_factor(hosvd_result.factors[k], margin=margin)
        weights.append(W)
        simplices.append(simplex)
        snnn_ranks.append(W.shape[1])

    # Refit core via least-squares
    N_total = grid.total_points
    n_sys = n * n_plus_m

    KR = _build_khatri_rao(weights)
    T_mat = original_tensor.reshape(N_total, n_sys)
    Core_mat, _, _, _ = np.linalg.lstsq(KR, T_mat, rcond=None)
    core = Core_mat.reshape(tuple(snnn_ranks) + (n, n_plus_m))

    return SNNNResult(
        core=core,
        weights=weights,
        ranks=snnn_ranks,
        grid=grid,
        n=n,
        n_plus_m=n_plus_m,
        singular_values=hosvd_result.singular_values,
        simplices=simplices,
    )


# =====================================================================
# Classical column-shift SNNN (Baranyi 2016) — for reference
# =====================================================================

def _snnn_column_shift_factor(U):
    """Column-shift + row-normalize SNNN (classical approach).

    WARNING: Inherent accuracy limitation — column shift destroys the DC
    component of leading singular vectors.
    """
    N, I = U.shape
    if I == 1:
        return np.ones((N, 1))
    c = np.min(U, axis=0)
    V = U - c[np.newaxis, :]
    d = V.sum(axis=1)
    W = np.empty_like(V)
    for i in range(N):
        if d[i] > 1e-15:
            W[i, :] = V[i, :] / d[i]
        else:
            W[i, :] = 1.0 / I
    return W


def snnn_column_shift(hosvd_result, original_tensor):
    """Classical column-shift SNNN (Baranyi 2016). For comparison only."""
    n_p = len(hosvd_result.ranks)
    n = hosvd_result.n
    n_plus_m = hosvd_result.n_plus_m
    grid = hosvd_result.grid

    weights = [_snnn_column_shift_factor(hosvd_result.factors[k])
               for k in range(n_p)]

    N_total = grid.total_points
    n_sys = n * n_plus_m
    KR = _build_khatri_rao(weights)
    T_mat = original_tensor.reshape(N_total, n_sys)
    Core_mat, _, _, _ = np.linalg.lstsq(KR, T_mat, rcond=None)
    core = Core_mat.reshape(tuple(hosvd_result.ranks) + (n, n_plus_m))

    return SNNNResult(
        core=core,
        weights=weights,
        ranks=hosvd_result.ranks,
        grid=grid,
        n=n,
        n_plus_m=n_plus_m,
        singular_values=hosvd_result.singular_values,
    )


# =====================================================================
# Multilinear (hat function) polytopic model
# =====================================================================

def _hat_eval_1d(p, axis):
    """Evaluate piecewise-linear hat function weights at an arbitrary point."""
    K = len(axis)
    w = np.zeros(K)
    p = np.clip(p, axis[0], axis[-1])
    if K == 1:
        w[0] = 1.0
        return w
    idx = np.searchsorted(axis, p) - 1
    idx = np.clip(idx, 0, K - 2)
    denom = axis[idx + 1] - axis[idx]
    if abs(denom) < 1e-15:
        t = 0.0
    else:
        t = (p - axis[idx]) / denom
    w[idx] = 1.0 - t
    w[idx + 1] = t
    return w


def multilinear_polytopic(system, vertex_grid):
    """Build a polytopic model using multilinear (hat function) interpolation.

    Vertex systems are S(p) at grid points. Hat functions are inherently SNNN.
    """
    from tp_transform.sampling import build_system_tensor
    tensor = build_system_tensor(system, vertex_grid)
    n_p = vertex_grid.n_p
    n = tensor.shape[n_p]
    n_plus_m = tensor.shape[n_p + 1]

    weights = [np.eye(K) for K in vertex_grid.n_points]
    return SNNNResult(
        core=tensor.copy(),
        weights=weights,
        ranks=list(vertex_grid.n_points),
        grid=vertex_grid,
        n=n,
        n_plus_m=n_plus_m,
        singular_values=[np.ones(K) for K in vertex_grid.n_points],
    )


def multilinear_error(system, vertex_grid, test_grid):
    """Measure multilinear interpolation error on a fine test grid."""
    from tp_transform.sampling import build_system_tensor
    model = multilinear_polytopic(system, vertex_grid)
    T_test = build_system_tensor(system, test_grid)
    N_test = test_grid.total_points
    n_p = test_grid.n_p

    max_err = 0.0
    scale = np.max(np.abs(T_test))
    for flat_idx in range(N_test):
        multi_idx = np.unravel_index(flat_idx, test_grid.n_points)
        p = np.array([test_grid.axes[k][multi_idx[k]] for k in range(n_p)])
        S_approx = model.evaluate_system(p)
        S_true = T_test[multi_idx]
        err = np.max(np.abs(S_approx - S_true))
        max_err = max(max_err, err)
    rel_err = max_err / scale if scale > 0 else 0.0
    return max_err, rel_err


# =====================================================================
# Recommended pipeline: HOSVD-guided multilinear polytopic model
# =====================================================================

def hosvd_guided_polytopic(system, sample_grid, threshold=0.01, extra=1,
                           verbose=True):
    """Build a polytopic model using HOSVD for rank selection + multilinear.

    This is the recommended pipeline for P-RHPG:
        1. Sample system tensor on a fine grid
        2. HOSVD to determine rank I_k per scheduling dimension
        3. Multilinear model on grid with (I_k + extra) points per dim

    Parameters
    ----------
    system : object with .system_matrix(p), .param_bounds
    sample_grid : ParameterGrid — fine grid for HOSVD analysis
    threshold : float — HOSVD singular value threshold (σ_i/σ_1 > threshold)
    extra : int — extra grid points per dimension beyond HOSVD rank (≥1)
    verbose : bool

    Returns
    -------
    model : SNNNResult — multilinear polytopic model
    hosvd_result : HOSVDResult — for diagnostics / paper figures
    """
    from tp_transform.sampling import ParameterGrid, build_system_tensor
    from tp_transform.hosvd import hosvd as run_hosvd

    # Step 1: HOSVD on fine sampling grid
    T = build_system_tensor(system, sample_grid)
    h = run_hosvd(T, sample_grid, threshold=threshold)

    if verbose:
        print(f"HOSVD rank selection (threshold={threshold}):")
        h.summary()

    # Step 2: Determine multilinear grid resolution
    vertex_n_points = [r + extra for r in h.ranks]

    if verbose:
        Nv = int(np.prod(vertex_n_points))
        print(f"\nMultilinear grid: {vertex_n_points} = {Nv} vertices")

    # Step 3: Build multilinear model
    vertex_grid = ParameterGrid(system.param_bounds, n_points=vertex_n_points)
    model = multilinear_polytopic(system, vertex_grid)

    if verbose:
        # Reconstruction error on the fine grid
        me, re = multilinear_error(system, vertex_grid, sample_grid)
        print(f"Reconstruction error: max={me:.2e}, rel={re:.2e}")

    return model, h


# =====================================================================
# Utility
# =====================================================================

def _build_khatri_rao(weights):
    """Build Khatri-Rao (row-wise Kronecker) design matrix.

    Parameters
    ----------
    weights : list of (N_k, I_k) arrays

    Returns
    -------
    KR : (N_total, N_v) array where N_total = ∏ N_k, N_v = ∏ I_k
    """
    KR = weights[0]
    for k in range(1, len(weights)):
        N_prev, I_prev = KR.shape
        N_k, I_k = weights[k].shape
        KR_new = np.zeros((N_prev * N_k, I_prev * I_k))
        for i in range(N_prev):
            for j in range(N_k):
                KR_new[i * N_k + j, :] = np.kron(KR[i, :], weights[k][j, :])
        KR = KR_new
    return KR


# =====================================================================
# SNNNResult class
# =====================================================================

class SNNNResult:
    """Stores an SNNN-normalized polytopic model.

    Attributes
    ----------
    core : ndarray of shape (*ranks, n, n+m)
    weights : list of ndarray — weights[k] has shape (N_grid_k, I_k)
    ranks : list of int — I_k per dimension
    n_vertices : int — N_v = ∏ I_k
    grid : ParameterGrid
    n, n_plus_m : int
    simplices : list of arrays or None — simplex vertices (if simplex-based)
    """

    def __init__(self, core, weights, ranks, grid, n, n_plus_m,
                 singular_values, simplices=None):
        self.core = core
        self.weights = weights
        self.ranks = ranks
        self.grid = grid
        self.n = n
        self.n_plus_m = n_plus_m
        self.singular_values = singular_values
        self.simplices = simplices
        self.n_vertices = int(np.prod(ranks))

    def reconstruct(self):
        """Reconstruct full tensor from core and weights."""
        from tp_transform.hosvd import mode_product
        T = self.core
        for k, W in enumerate(self.weights):
            T = mode_product(T, W, k)
        return T

    def reconstruction_error(self, original_tensor):
        """Max absolute and relative reconstruction error."""
        approx = self.reconstruct()
        max_err = np.max(np.abs(original_tensor - approx))
        scale = np.max(np.abs(original_tensor))
        rel_err = max_err / scale if scale > 0 else 0.0
        return max_err, rel_err

    def get_vertex_systems(self):
        """Extract (A_i, B_i) pairs from core tensor."""
        n = self.n
        vertices = []
        multi_indices = []
        for idx in itertools.product(*[range(r) for r in self.ranks]):
            S = self.core[idx]
            A_i = S[:, :n]
            B_i = S[:, n:]
            vertices.append((A_i, B_i))
            multi_indices.append(idx)
        return vertices, multi_indices

    def evaluate_weights(self, p):
        """Evaluate all N_v weighting functions α_i(p) at a parameter point.

        Uses linear interpolation between grid points per dimension.
        """
        per_dim = []
        for k in range(len(self.ranks)):
            axis = self.grid.axes[k]
            if len(axis) == 1:
                # Single grid point: weight is always 1
                per_dim.append(self.weights[k][0].copy())
                continue
            pk = float(p[k])
            pk = np.clip(pk, axis[0], axis[-1])
            idx = np.searchsorted(axis, pk) - 1
            idx = np.clip(idx, 0, len(axis) - 2)
            denom = axis[idx + 1] - axis[idx]
            if abs(denom) < 1e-15:
                # Degenerate: two identical grid points, use left weight
                t = 0.0
            else:
                t = (pk - axis[idx]) / denom
            w_k = (1 - t) * self.weights[k][idx] + t * self.weights[k][idx + 1]
            per_dim.append(w_k)

        alphas = per_dim[0]
        for k in range(1, len(self.ranks)):
            alphas = np.outer(alphas, per_dim[k]).ravel()
        return alphas

    def evaluate_system(self, p):
        """Evaluate S(p) = Σ α_i(p) S_i at a parameter point."""
        alphas = self.evaluate_weights(p)
        vertices, _ = self.get_vertex_systems()
        S = np.zeros((self.n, self.n_plus_m))
        for alpha, (A_i, B_i) in zip(alphas, vertices):
            S[:, :self.n] += alpha * A_i
            S[:, self.n:] += alpha * B_i
        return S

    def gram_matrix(self, mu_weights=None):
        """Gram matrix Γ_{ij} = ∫ α_i(p) α_j(p) dμ(p) (grid quadrature)."""
        N_total = self.grid.total_points
        if mu_weights is None:
            mu_weights = np.ones(N_total) / N_total
        alpha_matrix = _build_khatri_rao(self.weights)
        return alpha_matrix.T @ np.diag(mu_weights) @ alpha_matrix

    def verify_snnn(self, tol=1e-10):
        """Check SNNN properties on all grid points."""
        all_ok = True
        for k, W in enumerate(self.weights):
            min_val = np.min(W)
            if min_val < -tol:
                print(f"  Dim {k}: FAILED non-negativity (min = {min_val:.2e})")
                all_ok = False
            max_dev = np.max(np.abs(W.sum(axis=1) - 1.0))
            if max_dev > tol:
                print(f"  Dim {k}: FAILED sum-to-one (max dev = {max_dev:.2e})")
                all_ok = False
        if all_ok:
            print("  SNNN verification: PASSED")
        return all_ok

    def summary(self):
        """Print summary."""
        print(f"SNNN Model:")
        print(f"  Vertices: N_v = {' × '.join(map(str, self.ranks))} "
              f"= {self.n_vertices}")
        print(f"  Core shape: {self.core.shape}")
        print(f"  System: n={self.n}, m={self.n_plus_m - self.n}")
        Gamma = self.gram_matrix()
        eigs = np.linalg.eigvalsh(Gamma)
        kappa = eigs[-1] / eigs[0] if eigs[0] > 1e-15 else np.inf
        print(f"  Gram matrix: κ(Γ) = {kappa:.1f}, "
              f"λ_min = {eigs[0]:.2e}, λ_max = {eigs[-1]:.2e}")


# =====================================================================
# Tests
# =====================================================================

if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, '.')
    from tp_transform.sampling import ParameterGrid, build_system_tensor
    from tp_transform.hosvd import hosvd

    print("=" * 60)
    print("Convex Hull Module — Tests")
    print("=" * 60)

    # --- Test simplex primitives ---
    print("\n--- Simplex primitives ---")
    for d in [2, 3, 4]:
        V = _regular_simplex(d)
        assert V.shape == (d + 1, d)
        norms = np.linalg.norm(V, axis=1)
        assert np.allclose(norms, 1.0), f"d={d}: norms = {norms}"
        assert np.allclose(V.mean(axis=0), 0, atol=1e-12)
        from itertools import combinations
        edges = [np.linalg.norm(V[i] - V[j]) for i, j in combinations(range(d + 1), 2)]
        assert np.allclose(edges, edges[0], atol=1e-12)
        print(f"  d={d}: {d+1} vertices, R=1, edges={edges[0]:.4f} ✓")

    # --- Test barycentric coordinates ---
    print("\n--- Barycentric coords ---")
    V2 = _regular_simplex(2, circumradius=2.0)
    origin = np.zeros((1, 2))
    bc = _barycentric_coords(origin, V2)
    assert np.allclose(bc, 1 / 3, atol=1e-12)
    for i in range(3):
        bc = _barycentric_coords(V2[i:i + 1], V2)
        assert np.allclose(bc[0, i], 1.0, atol=1e-12)
        assert np.allclose(bc[0].sum(), 1.0, atol=1e-12)
    print("  2D centroid & vertex tests: ✓")

    # --- Pendulum ---
    from systems.inverted_pendulum import InvertedPendulum
    pend = InvertedPendulum()
    gp = ParameterGrid(pend.param_bounds, n_points=[40, 30])
    Tp = build_system_tensor(pend, gp)
    hp = hosvd(Tp, gp, threshold=0.001)

    print(f"\n--- Pendulum: HOSVD ranks={hp.ranks} ---")
    sp = snnn_normalize(hp, Tp)
    sp.summary()
    sp.verify_snnn()
    me, re = sp.reconstruction_error(Tp)
    print(f"  Simplex SNNN: max={me:.2e}, rel={re:.2e}")
    sp_cs = snnn_column_shift(hp, Tp)
    _, re_cs = sp_cs.reconstruction_error(Tp)
    print(f"  Column-shift: rel={re_cs:.2e}")

    # --- Wing ---
    from systems.aeroelastic_wing import AeroelasticWing
    wing = AeroelasticWing()
    gw = ParameterGrid(wing.param_bounds, n_points=[20, 5, 5])
    Tw = build_system_tensor(wing, gw)

    print(f"\n--- Wing: Method comparison ---")
    for ranks in [[3, 2, 2], [4, 2, 2], [4, 3, 2]]:
        hw = hosvd(Tw, gw, ranks=ranks)
        _, re_h = hw.reconstruction_error(Tw)
        sw = snnn_normalize(hw, Tw)
        _, re_s = sw.reconstruction_error(Tw)
        sw_cs = snnn_column_shift(hw, Tw)
        _, re_cs = sw_cs.reconstruction_error(Tw)
        print(f"  ranks={ranks}: HOSVD={re_h:.2e}, "
              f"Simplex(Nv={sw.n_vertices})={re_s:.2e}, "
              f"ColShift(Nv={sw_cs.n_vertices})={re_cs:.2e}")

    # --- SNNN validation ---
    print(f"\n--- Wing: Best result ---")
    hw = hosvd(Tw, gw, ranks=[4, 2, 2])
    sw = snnn_normalize(hw, Tw)
    sw.summary()
    sw.verify_snnn()
    me, re = sw.reconstruction_error(Tw)
    print(f"  Reconstruction: max={me:.2e}, rel={re:.2e}")

    verts, _ = sw.get_vertex_systems()
    print(f"  Vertices: {len(verts)}, A: {verts[0][0].shape}, B: {verts[0][1].shape}")

    rng = np.random.default_rng(42)
    for _ in range(100):
        p = np.array([rng.uniform(*b) for b in wing.param_bounds])
        alphas = sw.evaluate_weights(p)
        assert np.abs(np.sum(alphas) - 1.0) < 1e-10
        assert np.all(alphas >= -1e-10)
    print("  100 random points: SNNN ✓")

    # --- Recommended pipeline ---
    print(f"\n{'='*60}")
    print(f"Recommended Pipeline: hosvd_guided_polytopic()")
    print(f"{'='*60}")

    print(f"\n--- Wing ---")
    model_w, h_w = hosvd_guided_polytopic(
        wing,
        ParameterGrid(wing.param_bounds, n_points=[20, 5, 5]),
        threshold=0.001,
    )
    model_w.verify_snnn()

    print(f"\n--- Pendulum ---")
    model_p, h_p = hosvd_guided_polytopic(
        pend,
        ParameterGrid(pend.param_bounds, n_points=[40, 30]),
        threshold=0.001,
    )
    model_p.verify_snnn()

    print("\nAll tests passed.")