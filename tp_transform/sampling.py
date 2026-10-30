"""
Parameter Space Sampling and System Tensor Construction
========================================================

Grids the scheduling parameter domain Ω and evaluates S(p) = [A(p) B(p)]
at each grid point to form the system tensor for HOSVD decomposition.

Supports 1D and 2D scheduling parameter spaces.
"""

import numpy as np


class ParameterGrid:
    """Uniform grid over a rectangular scheduling parameter domain.

    Parameters
    ----------
    bounds : (n_p, 2) array — [[lo_1, hi_1], ..., [lo_d, hi_d]]
    n_points : int or list of int — points per dimension
    """

    def __init__(self, bounds, n_points=50):
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_p = self.bounds.shape[0]

        if np.isscalar(n_points):
            self.n_points = [n_points] * self.n_p
        else:
            assert len(n_points) == self.n_p
            self.n_points = list(n_points)

        # 1D grids along each scheduling dimension
        self.axes = [np.linspace(self.bounds[d, 0], self.bounds[d, 1], self.n_points[d]) for d in range(self.n_p)]

        # Full grid via meshgrid (ij indexing for tensor ordering)
        if self.n_p == 1:
            self.grid_points = self.axes[0].reshape(-1, 1)
        else:
            meshes = np.meshgrid(*self.axes, indexing="ij")
            self.grid_points = np.stack([m.ravel() for m in meshes], axis=-1)
            self.grid_shape = tuple(self.n_points)

    @property
    def total_points(self):
        return int(np.prod(self.n_points))

    def __repr__(self):
        return f"ParameterGrid(n_p={self.n_p}, points_per_dim={self.n_points}, total={self.total_points})"


def build_system_tensor(system, grid, discrete=True):
    """Evaluate S(p) = [A(p) B(p)] on a parameter grid to form the system tensor.

    Parameters
    ----------
    system : object
        Must have a `system_matrix(p, discrete)` method returning (n, n+m) array.
    grid : ParameterGrid
    discrete : bool

    Returns
    -------
    tensor : ndarray
        Shape (*grid.n_points, n, n+m).
        For 1D scheduling: (N_grid, n, n+m).
        For 2D scheduling: (N1, N2, n, n+m).
    """
    # Probe dimensions from the first grid point
    S0 = system.system_matrix(grid.grid_points[0], discrete=discrete)
    n, n_plus_m = S0.shape

    # Allocate tensor
    tensor_shape = (*tuple(grid.n_points), n, n_plus_m)
    tensor = np.empty(tensor_shape)

    # Fill — iterate over flattened grid, then reshape
    flat = tensor.reshape(-1, n, n_plus_m)
    for k in range(grid.total_points):
        p = grid.grid_points[k]
        flat[k] = system.system_matrix(p, discrete=discrete)

    return tensor


# ── Quick test ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path

    # Ensure project root (parent of this file's directory) is on sys.path
    _ROOT = _Path(__file__).resolve().parents[1]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))

    print("=" * 50)
    print("Sampling Module — Tests")
    print("=" * 50)

    # --- Test grid construction ---
    g1 = ParameterGrid(bounds=np.array([[5.0, 35.0]]), n_points=60)
    print(f"\n1D grid: {g1}")
    assert g1.grid_points.shape == (60, 1)

    g2 = ParameterGrid(bounds=np.array([[-1.0, 1.0], [-2.0, 2.0]]), n_points=[30, 20])
    print(f"2D grid: {g2}")
    assert g2.grid_points.shape == (600, 2)
    assert g2.grid_shape == (30, 20)

    g3 = ParameterGrid(bounds=np.array([[5.0, 35.0], [-0.1, 0.1], [-0.3, 0.3]]), n_points=[20, 5, 5])
    print(f"3D grid: {g3}")
    assert g3.grid_points.shape == (500, 3)
    assert g3.grid_shape == (20, 5, 5)

    # --- Test with inverted pendulum (2D scheduling) ---
    from systems.inverted_pendulum import InvertedPendulum

    pend = InvertedPendulum()
    gp = ParameterGrid(pend.param_bounds, n_points=[40, 30])
    Tp = build_system_tensor(pend, gp)
    print(f"\nPendulum tensor: {Tp.shape}  (expected (40, 30, 4, 5))")
    assert Tp.shape == (40, 30, 4, 5)

    p_test = np.array([gp.axes[0][20], gp.axes[1][15]])
    err = np.max(np.abs(Tp[20, 15] - pend.system_matrix(p_test)))
    print(f"  Spot check: {err:.2e}")
    assert err < 1e-14

    # --- Test with aeroelastic wing (3D scheduling) ---
    from systems.aeroelastic_wing import AeroelasticWing

    wing = AeroelasticWing()
    gw = ParameterGrid(wing.param_bounds, n_points=[20, 5, 5])
    Tw = build_system_tensor(wing, gw)
    print(f"\nWing tensor:     {Tw.shape}  (expected (20, 5, 5, 4, 5))")
    assert Tw.shape == (20, 5, 5, 4, 5)

    p_test = np.array([gw.axes[0][10], gw.axes[1][3], gw.axes[2][2]])
    err = np.max(np.abs(Tw[10, 3, 2] - wing.system_matrix(p_test)))
    print(f"  Spot check: {err:.2e}")
    assert err < 1e-14

    print("\nAll tests passed.")
