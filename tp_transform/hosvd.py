"""
Higher-Order Singular Value Decomposition (HOSVD)
=================================================

Decomposes the system tensor S(p) sampled on a grid into:

    S ≈ Core  ×₁ U₁  ×₂ U₂  ×₃ U₃  ...

where ×ₙ is the n-mode product, Uₙ holds the left singular vectors
along scheduling dimension n, and Core is the compressed core tensor.

The truncation rank Iₙ along each dimension is chosen by discarding
singular values below a relative threshold. The total number of
vertices is N_v = ∏ Iₙ.

After HOSVD, the weighting vectors (rows of Uₙ evaluated on the grid)
and vertex system matrices (entries of Core) are passed to convex_hull.py
for SNNN normalization.

Reference:
    De Lathauwer, De Moor, Vandewalle, "A Multilinear Singular Value
    Decomposition", SIAM J. Matrix Anal. Appl. 21(4), 2000.
"""

import numpy as np


def unfold(tensor, mode):
    """Mode-n unfolding (matricization) of a tensor.

    Parameters
    ----------
    tensor : ndarray of shape (I_0, I_1, ..., I_{N-1})
    mode : int — dimension to unfold along

    Returns
    -------
    matrix : ndarray of shape (I_mode, prod of other dims)
    """
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def mode_product(tensor, matrix, mode):
    """N-mode product: tensor ×_mode matrix.

    Multiplies matrix (J × I_mode) along dimension `mode` of the tensor,
    replacing that dimension's size from I_mode to J.

    Parameters
    ----------
    tensor : ndarray
    matrix : (J, I_mode) array
    mode : int

    Returns
    -------
    result : ndarray with shape[mode] replaced by J
    """
    result = np.tensordot(matrix, tensor, axes=([1], [mode]))
    return np.moveaxis(result, 0, mode)


class HOSVDResult:
    """Stores the output of an HOSVD decomposition.

    Attributes
    ----------
    core : ndarray
        Core tensor of shape (I_1, ..., I_d, n, n+m).
    factors : list of ndarray
        factors[k] has shape (N_grid_k, I_k) — columns are left singular
        vectors along scheduling dimension k, evaluated on the grid.
    singular_values : list of ndarray
        Full singular value spectrum along each scheduling dimension.
    ranks : list of int
        Truncation rank I_k for each dimension.
    n_vertices : int
        Total vertex count N_v = prod(ranks).
    grid : ParameterGrid
        The grid used for sampling (stored for convenience).
    n : int
        State dimension.
    n_plus_m : int
        n + m (columns of the system matrix).
    """

    def __init__(self, core, factors, singular_values, ranks, grid, n, n_plus_m):
        self.core = core
        self.factors = factors
        self.singular_values = singular_values
        self.ranks = ranks
        self.grid = grid
        self.n = n
        self.n_plus_m = n_plus_m
        self.n_vertices = int(np.prod(ranks))

    def reconstruct(self):
        """Reconstruct the full tensor from core and factors.

        Returns
        -------
        tensor : ndarray of shape (*grid.n_points, n, n+m)
        """
        T = self.core
        for k, U in enumerate(self.factors):
            T = mode_product(T, U, k)
        return T

    def reconstruction_error(self, original_tensor):
        """Max absolute reconstruction error.

        Parameters
        ----------
        original_tensor : ndarray of shape (*grid.n_points, n, n+m)

        Returns
        -------
        max_err : float
        rel_err : float — max_err / max|original|
        """
        approx = self.reconstruct()
        max_err = np.max(np.abs(original_tensor - approx))
        scale = np.max(np.abs(original_tensor))
        rel_err = max_err / scale if scale > 0 else 0.0
        return max_err, rel_err

    def get_vertex_systems(self):
        """Extract the N_v vertex system matrices from the core tensor.

        Returns
        -------
        vertices : list of (n, n+m) arrays, length N_v
        multi_indices : list of tuples, the corresponding index into core
        """
        n_p = len(self.ranks)
        # Iterate over all scheduling index combinations
        idx_ranges = [range(r) for r in self.ranks]
        import itertools

        vertices = []
        multi_indices = []
        for idx in itertools.product(*idx_ranges):
            S_vertex = self.core[idx]  # shape (n, n+m)
            vertices.append(S_vertex)
            multi_indices.append(idx)
        return vertices, multi_indices

    def summary(self):
        """Print a summary of the decomposition."""
        print(f"HOSVD Summary:")
        print(f"  Scheduling dims: {len(self.ranks)}")
        for k, (r, sv) in enumerate(zip(self.ranks, self.singular_values)):
            total = len(sv)
            ratio = sv[r - 1] / sv[0] if sv[0] > 0 else 0
            print(f"  Dim {k}: grid={self.factors[k].shape[0]}, rank={r}/{total}, σ_{r}/σ_1={ratio:.2e}")
        print(f"  Vertices: N_v = {' × '.join(map(str, self.ranks))} = {self.n_vertices}")
        print(f"  Core shape: {self.core.shape}")


def hosvd(tensor, grid, ranks=None, threshold=0.01):
    """Compute the (truncated) HOSVD of a system tensor.

    Parameters
    ----------
    tensor : ndarray of shape (*n_points, n, n+m)
        System tensor from build_system_tensor.
    grid : ParameterGrid
        The parameter grid used to sample the tensor.
    ranks : list of int, optional
        Explicit truncation ranks [I_1, ..., I_d]. If None, determined
        automatically from threshold.
    threshold : float
        Relative singular value threshold: keep σ_i if σ_i / σ_1 > threshold.
        Only used when ranks is None.

    Returns
    -------
    result : HOSVDResult
    """
    n_p = grid.n_p
    n_grid = list(grid.n_points)

    # Separate scheduling dimensions from system matrix dimensions
    # tensor shape: (*n_grid, n, n_plus_m)
    n = tensor.shape[n_p]
    n_plus_m = tensor.shape[n_p + 1]

    # Flatten the system matrix dimensions into one for the HOSVD
    # working shape: (*n_grid, n * n_plus_m)
    flat_shape = tuple(n_grid) + (n * n_plus_m,)
    T_flat = tensor.reshape(flat_shape)

    factors = []
    all_sv = []
    actual_ranks = []

    for k in range(n_p):
        # Mode-k unfolding: (n_grid_k, everything_else)
        M = unfold(T_flat, k)

        # Economy SVD
        U, s, _ = np.linalg.svd(M, full_matrices=False)

        all_sv.append(s.copy())

        # Determine rank
        if ranks is not None:
            r = ranks[k]
        else:
            # Keep singular values above threshold * σ_1
            above = s / s[0] > threshold
            r = max(1, int(np.sum(above)))

        r = min(r, len(s), M.shape[0])  # can't exceed grid size or rank
        actual_ranks.append(r)

        # Truncate: keep first r columns
        factors.append(U[:, :r])

    # Compute core tensor: project T_flat onto all factor bases
    # Core = T_flat ×₁ U₁ᵀ ×₂ U₂ᵀ ... ×_d U_dᵀ
    core_flat = T_flat
    for k in range(n_p):
        core_flat = mode_product(core_flat, factors[k].T, k)

    # Reshape core back to (*ranks, n, n_plus_m)
    core = core_flat.reshape(tuple(actual_ranks) + (n, n_plus_m))

    return HOSVDResult(core, factors, all_sv, actual_ranks, grid, n, n_plus_m)


# ── Quick test ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys

    _sys.path.insert(0, ".")
    from sampling import ParameterGrid, build_system_tensor

    print("=" * 50)
    print("HOSVD Module — Tests")
    print("=" * 50)

    # --- Test primitives ---
    T = np.arange(24).reshape(2, 3, 4).astype(float)
    M0 = unfold(T, 0)
    assert M0.shape == (2, 12), f"Expected (2,12), got {M0.shape}"
    M1 = unfold(T, 1)
    assert M1.shape == (3, 8), f"Expected (3,8), got {M1.shape}"
    print("\nUnfold: OK")

    A = np.random.randn(5, 2)
    T2 = np.random.randn(3, 2, 4)
    R = mode_product(T2, A, 1)
    assert R.shape == (3, 5, 4)
    print("Mode product: OK")

    # --- Test with inverted pendulum (2D) ---
    from inverted_pendulum import InvertedPendulum

    pend = InvertedPendulum()
    gp = ParameterGrid(pend.param_bounds, n_points=[40, 30])
    Tp = build_system_tensor(pend, gp)

    result_p = hosvd(Tp, gp, threshold=0.001)
    result_p.summary()
    max_err, rel_err = result_p.reconstruction_error(Tp)
    print(f"  Reconstruction: max={max_err:.2e}, rel={rel_err:.2e}")
    assert rel_err < 0.01, f"Reconstruction too poor: {rel_err}"

    # --- Test with aeroelastic wing (3D) ---
    from aeroelastic_wing import AeroelasticWing

    wing = AeroelasticWing()
    gw = ParameterGrid(wing.param_bounds, n_points=[20, 5, 5])
    Tw = build_system_tensor(wing, gw)

    result_w = hosvd(Tw, gw, threshold=0.001)
    result_w.summary()
    max_err, rel_err = result_w.reconstruction_error(Tw)
    print(f"  Reconstruction: max={max_err:.2e}, rel={rel_err:.2e}")
    assert rel_err < 0.01

    # Check vertex extraction
    verts, idxs = result_w.get_vertex_systems()
    print(f"  Vertices: {len(verts)}, each shape {verts[0].shape}")
    assert len(verts) == result_w.n_vertices
    assert verts[0].shape == (4, 5)

    # --- Test exact recovery with explicit ranks ---
    print("\n--- Exact recovery (full rank) ---")
    result_full = hosvd(Tw, gw, ranks=list(gw.n_points))
    max_err, rel_err = result_full.reconstruction_error(Tw)
    print(f"  Full rank reconstruction: max={max_err:.2e}, rel={rel_err:.2e}")
    assert max_err < 1e-10, "Full-rank HOSVD should be exact"

    print("\nAll tests passed.")
