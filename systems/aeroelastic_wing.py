"""
2-DoF Aeroelastic Wing Section with Structural Uncertainty
==========================================================

Classical pitch-plunge typical section with quasi-steady aerodynamics
and parametric uncertainty in the structural stiffness and damping.

Scheduling parameters (3D):
    p = (V, delta_K, delta_C)

    V       : airspeed [m/s]                — measured, varies in flight
    delta_K : fractional stiffness shift     — uncertain, in [-eps_K, eps_K]
    delta_C : fractional damping shift       — uncertain, in [-eps_C, eps_C]

The perturbed structural matrices are:
    K_s(delta_K) = K_s0 * (1 + delta_K)
    C_s(delta_C) = C_s0 * (1 + delta_C)

This gives an LPV system with 3 scheduling dimensions, following the
Takarics & Vanek (2021) approach for the FLEXOP aircraft. The TP model
transformation produces N_v = I_1 x I_2 x I_3 vertices from the HOSVD
truncation ranks along each dimension.

State:  x = [h, dh/dt, alpha, dalpha/dt]^T  (n=4)
Input:  u = beta (control surface deflection) (m=1)

References:
    - Takarics & Vanek, Asian J. Control 2021 (FLEXOP TP model)
    - Fung, "An Introduction to the Theory of Aeroelasticity", Dover 1969
    - Hodges & Pierce, "Intro to Structural Dynamics & Aeroelasticity", 2011
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


class AeroelasticWing:
    """2-DoF pitch-plunge airfoil with 3D scheduling (V, delta_K, delta_C).

    Parameters
    ----------
    b         : float   Semi-chord [m]
    a         : float   Elastic axis position (nondim by b, from midchord)
    m_wing    : float   Section mass per unit span [kg/m]
    x_alpha   : float   CG offset from EA (nondim by b)
    r_alpha   : float   Radius of gyration about EA (nondim by b)
    omega_h   : float   Nominal plunge natural frequency [rad/s]
    omega_a   : float   Nominal pitch natural frequency [rad/s]
    zeta_h    : float   Nominal plunge damping ratio
    zeta_a    : float   Nominal pitch damping ratio
    rho_air   : float   Air density [kg/m^3]
    C_La      : float   Lift curve slope [1/rad]
    C_Lb      : float   Control surface lift effectiveness [1/rad]
    eps_K     : float   Stiffness uncertainty bound (fractional, e.g. 0.1 = ±10%)
    eps_C     : float   Damping uncertainty bound (fractional, e.g. 0.3 = ±30%)
    Ts        : float   Sampling time [s]
    V_min, V_max : float  Airspeed range [m/s]
    """

    def __init__(
        self,
        b=0.3,
        a=-0.35,
        m_wing=10.0,
        x_alpha=0.05,
        r_alpha=0.5,
        omega_h=30.0,
        omega_a=55.0,
        zeta_h=0.02,
        zeta_a=0.05,
        rho_air=1.225,
        C_La=2 * np.pi,
        C_Lb=1.0,
        eps_K=0.10,
        eps_C=0.30,
        Ts=0.002,
        V_min=5.0,
        V_max=35.0,
    ):
        self.b = b
        self.a = a
        self.m_wing = m_wing
        self.x_alpha = x_alpha
        self.r_alpha = r_alpha
        self.rho_air = rho_air
        self.C_La = C_La
        self.C_Lb = C_Lb
        self.eps_K = eps_K
        self.eps_C = eps_C
        self.Ts = Ts
        self.V_min = V_min
        self.V_max = V_max

        # Dimensions
        self.n = 4
        self.n_u = 1
        self.n_p = 3  # (V, delta_K, delta_C)

        # Derived structural parameters (nominal)
        self.S_alpha = m_wing * x_alpha * b
        self.I_alpha = m_wing * r_alpha**2 * b**2
        self.k_h0 = m_wing * omega_h**2
        self.k_a0 = self.I_alpha * omega_a**2
        self.c_h0 = 2 * zeta_h * m_wing * omega_h
        self.c_a0 = 2 * zeta_a * self.I_alpha * omega_a

        # Mass matrix (constant, independent of scheduling)
        self.M_s = np.array(
            [
                [m_wing, self.S_alpha],
                [self.S_alpha, self.I_alpha],
            ]
        )
        self.M_s_inv = np.linalg.inv(self.M_s)

        # Scheduling domain
        self.param_bounds = np.array(
            [
                [V_min, V_max],  # p1 = V
                [-eps_K, eps_K],  # p2 = delta_K
                [-eps_C, eps_C],  # p3 = delta_C
            ]
        )

    # ── Structural matrices (depend on delta_K, delta_C) ─────

    def _structural_matrices(self, delta_K, delta_C):
        """Perturbed structural stiffness and damping.

        K_s = diag(k_h, k_a) * (1 + delta_K)
        C_s = diag(c_h, c_a) * (1 + delta_C)
        """
        K_s = np.diag([self.k_h0, self.k_a0]) * (1.0 + delta_K)
        C_s = np.diag([self.c_h0, self.c_a0]) * (1.0 + delta_C)
        return K_s, C_s

    # ── Aerodynamic matrices (depend on V) ───────────────────

    def _aero_matrices(self, V):
        """Quasi-steady aero: damping C_a(V), stiffness K_a(V), control B_a(V)."""
        rho, b, a_ea = self.rho_air, self.b, self.a
        C_La, C_Lb = self.C_La, self.C_Lb
        d = (a_ea + 0.5) * b  # aero center to EA distance

        C_a = (
            rho
            * V
            * b
            * C_La
            * np.array(
                [
                    [1.0, b * (0.5 - a_ea)],
                    [-d, -d * b * (0.5 - a_ea)],
                ]
            )
        )

        K_a = (
            rho
            * V**2
            * b
            * C_La
            * np.array(
                [
                    [0.0, 1.0],
                    [0.0, -d],
                ]
            )
        )

        B_a = (
            rho
            * V**2
            * b
            * C_Lb
            * np.array(
                [
                    [-1.0],
                    [d],
                ]
            )
        )

        return C_a, K_a, B_a

    # ── Continuous-time state-space ──────────────────────────

    def A_continuous(self, p):
        """Continuous-time A(p) where p = (V, delta_K, delta_C).

        Parameters
        ----------
        p : (3,) array

        Returns
        -------
        A : (4, 4) array
        """
        V, dK, dC = float(p[0]), float(p[1]), float(p[2])

        K_s, C_s = self._structural_matrices(dK, dC)
        C_a, K_a, _ = self._aero_matrices(V)

        Mi = self.M_s_inv
        A = np.zeros((4, 4))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 0:2] = -Mi @ (K_s + K_a)
        A[2:4, 2:4] = -Mi @ (C_s + C_a)
        return A

    def B_continuous(self, p):
        """Continuous-time B(p). Depends on V only.

        Parameters
        ----------
        p : (3,) array

        Returns
        -------
        B : (4, 1) array
        """
        V = float(p[0])
        _, _, B_a = self._aero_matrices(V)

        B = np.zeros((4, 1))
        B[2:4, :] = self.M_s_inv @ B_a
        return B

    # ── Discrete-time (ZOH) ──────────────────────────────────

    def A_discrete(self, p):
        """Discrete A_d(p) = expm(A_c(p) * Ts)."""
        return expm(self.A_continuous(p) * self.Ts)

    def B_discrete(self, p):
        """Discrete B_d(p) via augmented matrix exponential."""
        Ac = self.A_continuous(p)
        Bc = self.B_continuous(p)
        n = self.n
        aug = np.zeros((n + self.n_u, n + self.n_u))
        aug[:n, :n] = Ac * self.Ts
        aug[:n, n:] = Bc * self.Ts
        return expm(aug)[:n, n:]

    # ── System tensor interface ──────────────────────────────

    def system_matrix(self, p, discrete=True):
        """Return S(p) = [A(p) B(p)].

        Parameters
        ----------
        p : (3,) array — (V, delta_K, delta_C)
        discrete : bool

        Returns
        -------
        S : (4, 5) array
        """
        if discrete:
            return np.hstack([self.A_discrete(p), self.B_discrete(p)])
        else:
            return np.hstack([self.A_continuous(p), self.B_continuous(p)])

    # ── Dynamics simulation ──────────────────────────────────

    def continuous_dynamics(self, x, u, p):
        """dx/dt = A(p) x + B(p) u."""
        return self.A_continuous(p) @ x + self.B_continuous(p).flatten() * float(u)

    def simulate_step(self, x, u, p):
        """One discrete step via RK4 (true dynamics)."""
        f = lambda t, s: self.continuous_dynamics(s, u, p)
        sol = solve_ivp(f, [0, self.Ts], x, method="RK45", rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]

    # ── Flutter analysis ─────────────────────────────────────

    def flutter_speed(self, delta_K=0.0, delta_C=0.0, n_points=500):
        """Find flutter speed at given uncertainty values.

        Returns
        -------
        V_flutter : float or None
        """
        V_vals = np.linspace(self.V_min, self.V_max, n_points)
        prev_re = -np.inf

        for V in V_vals:
            A = self.A_continuous([V, delta_K, delta_C])
            max_re = np.max(np.linalg.eigvals(A).real)
            if prev_re < 0 and max_re >= 0:
                return V
            prev_re = max_re
        return None

    # ── Validation ───────────────────────────────────────────

    def validate_lpv(self, n_tests=200, tol=1e-10):
        """Verify A(p)x + B(p)u = continuous_dynamics(x, u, p)."""
        max_err = 0.0
        rng = np.random.default_rng(42)

        for _ in range(n_tests):
            p = np.array(
                [
                    rng.uniform(self.V_min, self.V_max),
                    rng.uniform(-self.eps_K, self.eps_K),
                    rng.uniform(-self.eps_C, self.eps_C),
                ]
            )
            x = rng.standard_normal(4) * np.array([0.01, 0.1, 0.05, 0.5])
            u = rng.uniform(-0.1, 0.1)

            f1 = self.continuous_dynamics(x, u, p)
            f2 = self.A_continuous(p) @ x + self.B_continuous(p).flatten() * u
            max_err = max(max_err, np.max(np.abs(f1 - f2)))

        passed = max_err < tol
        print(f"LPV validation: max error = {max_err:.2e}  ({'PASSED' if passed else 'FAILED'})")
        return max_err

    def validate_discretization(self, n_tests=50, tol=1e-2):
        """Check ZOH discrete step vs RK4."""
        max_err = 0.0
        rng = np.random.default_rng(123)

        for _ in range(n_tests):
            p = np.array(
                [
                    rng.uniform(self.V_min, self.V_max),
                    rng.uniform(-self.eps_K, self.eps_K),
                    rng.uniform(-self.eps_C, self.eps_C),
                ]
            )
            x = rng.standard_normal(4) * np.array([0.005, 0.05, 0.02, 0.2])
            u = rng.uniform(-0.05, 0.05)

            x_rk4 = self.simulate_step(x, u, p)
            x_zoh = self.A_discrete(p) @ x + self.B_discrete(p).flatten() * u
            max_err = max(max_err, np.max(np.abs(x_rk4 - x_zoh)))

        passed = max_err < tol
        print(f"ZOH validation: max error = {max_err:.2e}  (Ts={self.Ts}, {'PASSED' if passed else 'FAILED'})")
        return max_err


# ── Quick test ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("2-DoF Aeroelastic Wing — 3D Scheduling (V, δK, δC)")
    print("=" * 60)

    sys = AeroelasticWing()
    print(f"\nState dim: {sys.n},  Input dim: {sys.n_u},  Scheduling dim: {sys.n_p}")
    print("Param bounds:")
    labels = ["V [m/s]", "delta_K", "delta_C"]
    for i, lab in enumerate(labels):
        print(f"  {lab:10s}: [{sys.param_bounds[i, 0]:+.3f}, {sys.param_bounds[i, 1]:+.3f}]")
    print(f"Sampling time: Ts = {sys.Ts}")

    print("\n--- LPV identity check ---")
    sys.validate_lpv()

    print("\n--- ZOH discretization check ---")
    sys.validate_discretization()

    # Flutter across uncertainty corners
    print("\n--- Flutter speed vs structural uncertainty ---")
    corners = [
        (0.0, 0.0, "nominal"),
        (-sys.eps_K, -sys.eps_C, "soft/low-damp"),
        (+sys.eps_K, +sys.eps_C, "stiff/high-damp"),
        (-sys.eps_K, +sys.eps_C, "soft/high-damp"),
        (+sys.eps_K, -sys.eps_C, "stiff/low-damp"),
    ]
    for dK, dC, label in corners:
        Vf = sys.flutter_speed(delta_K=dK, delta_C=dC)
        Vf_str = f"{Vf:.2f} m/s" if Vf else "none in range"
        print(f"  dK={dK:+.2f}, dC={dC:+.2f} ({label:16s}): V_flutter = {Vf_str}")

    # B variation with airspeed
    print("\n--- B(p) variation with airspeed ---")
    for V in [10.0, 20.0, 30.0]:
        p = [V, 0.0, 0.0]
        B = sys.B_continuous(p).flatten()
        print(f"  V={V:5.1f}: B = [{B[2]:8.3f}, {B[3]:8.3f}]  (|B|={np.linalg.norm(B):.3f})")

    # Discrete stability
    print("\n--- Discrete spectral radius (nominal) ---")
    for V in [10.0, 20.0, 30.0]:
        p = [V, 0.0, 0.0]
        rho = np.max(np.abs(np.linalg.eigvals(sys.A_discrete(p))))
        print(f"  V={V:5.1f}: rho(A_d) = {rho:.6f}  ({'UNSTABLE' if rho > 1 else 'stable'})")

    # Expected tensor shape
    print("\n--- Expected tensor for TP transform ---")
    n_grid = [20, 5, 5]
    print(f"  Grid: {n_grid[0]} x {n_grid[1]} x {n_grid[2]} = {np.prod(n_grid)} points")
    print(f"  Tensor shape: ({n_grid[0]}, {n_grid[1]}, {n_grid[2]}, 4, 5)")
    print("  Vertices: N_v = I1 x I2 x I3  (from HOSVD truncation)")
