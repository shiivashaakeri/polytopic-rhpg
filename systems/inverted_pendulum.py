"""
Inverted Pendulum on Cart — Nonlinear System with Exact qLPV Form
=================================================================

Nonlinear dynamics:
    (M+m) x_dd + m l theta_dd cos(theta) - m l theta_d^2 sin(theta) = F
    m l x_dd cos(theta) + m l^2 theta_dd - m g l sin(theta) = 0

State:  x = [x, x_dot, theta, theta_dot]^T  (n=4)
Input:  u = F (force on cart)                 (m=1)

Scheduling parameters (state-dependent, 2D):
    p = (theta, theta_dot)  in  Omega = [-theta_max, theta_max] x [-omega_max, omega_max]

qLPV form:  dx/dt = A(p) x + B(p) u   (exact, no approximation)

Key identity:  sin(theta) = sinc(theta) * theta,  where sinc(t) = sin(t)/t
This lets us write the gravitational terms as A(p) * x (linear in state).
The centripetal term theta_dot^2 * sin(theta) = [theta_dot * sin(theta)] * theta_dot
requires theta_dot as a second scheduling parameter.
"""

import numpy as np
from scipy.integrate import solve_ivp


def _sinc(t):
    """sin(t)/t with removable singularity at t=0. Works for scalars and arrays."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    out = np.ones_like(t)
    nz = np.abs(t) > 1e-12
    out[nz] = np.sin(t[nz]) / t[nz]
    return float(out[0]) if scalar else out


class InvertedPendulum:
    """Inverted pendulum on cart with exact qLPV representation.

    Parameters
    ----------
    M : float          Cart mass [kg]
    m : float          Pendulum mass [kg]
    l : float          Pendulum half-length [m]
    g : float          Gravitational acceleration [m/s^2]
    Ts : float         Sampling time for discretization [s]
    theta_max : float  Max angle for scheduling domain [rad]
    omega_max : float  Max angular rate for scheduling domain [rad/s]
    """

    def __init__(self, M=1.0, m=0.3, l=0.5, g=9.81, Ts=0.02, theta_max=np.pi / 3, omega_max=2.0):
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.Ts = Ts

        # Dimensions
        self.n = 4  # state dimension
        self.n_u = 1  # input dimension
        self.n_p = 2  # scheduling parameter dimension

        # Scheduling domain Omega
        self.theta_max = theta_max
        self.omega_max = omega_max
        self.param_bounds = np.array(
            [
                [-theta_max, theta_max],  # p1 = theta
                [-omega_max, omega_max],  # p2 = theta_dot
            ]
        )

    # ── Helpers ──────────────────────────────────────────────

    def _D(self, theta):
        """Mass matrix determinant: D(theta) = m l^2 (M + m sin^2(theta))."""
        return self.m * self.l**2 * (self.M + self.m * np.sin(theta) ** 2)

    # ── Nonlinear dynamics ───────────────────────────────────

    def nonlinear_dynamics(self, x, u):
        """Continuous-time nonlinear: dx/dt = f(x, u).

        Parameters
        ----------
        x : (4,) array — [x, x_dot, theta, theta_dot]
        u : float or (1,) array — force on cart

        Returns
        -------
        dx : (4,) array
        """
        _, x_d, th, th_d = x
        F = float(np.squeeze(u))
        M, m, l, g = self.M, self.m, self.l, self.g

        s, c = np.sin(th), np.cos(th)
        D = m * l**2 * (M + m * s**2)

        # From inverting the 2x2 mass matrix
        x_dd = (m * l**2 * F + m**2 * l**3 * th_d**2 * s - m**2 * g * l**2 * c * s) / D
        th_dd = (-m * l * c * F - m**2 * l**2 * c * th_d**2 * s + (M + m) * m * g * l * s) / D

        return np.array([x_d, x_dd, th_d, th_dd])

    def nonlinear_step(self, x, u):
        """One discrete step of the true nonlinear dynamics (RK4).

        Parameters
        ----------
        x : (4,) array
        u : float — held constant over [0, Ts]

        Returns
        -------
        x_next : (4,) array
        """
        f = lambda t, s: self.nonlinear_dynamics(s, u)
        sol = solve_ivp(f, [0, self.Ts], x, method="RK45", rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]

    # ── Scheduling parameter ─────────────────────────────────

    def scheduling_param(self, x):
        """Extract p = (theta, theta_dot) from state.

        Parameters
        ----------
        x : (4,) array

        Returns
        -------
        p : (2,) array
        """
        return np.array([x[2], x[3]])

    # ── Exact qLPV matrices (continuous-time) ────────────────

    def A_continuous(self, p):
        """Continuous-time A(p) such that f(x,u) = A(p)x + B(p)u exactly.

        The qLPV identity uses:
          sin(theta) = sinc(theta) * theta    (absorbs into A * x_3)
          theta_dot^2 sin(theta) = [theta_dot sin(theta)] * theta_dot
                                                (absorbs into A * x_4)

        Parameters
        ----------
        p : (2,) array — [theta, theta_dot]

        Returns
        -------
        A : (4, 4) array
        """
        th, th_d = p
        M, m, l, g = self.M, self.m, self.l, self.g

        s = np.sin(th)
        c = np.cos(th)
        sc = _sinc(th)  # sin(th)/th
        D = m * l**2 * (M + m * s**2)

        A = np.zeros((4, 4))
        A[0, 1] = 1.0
        A[2, 3] = 1.0

        # x_dd terms (row 1):
        #   from gravity:  -m^2 g l^2 cos(th) sin(th) / D
        #     = -m^2 g l^2 cos(th) sinc(th) / D  *  theta  (= x_3)
        A[1, 2] = -(m**2) * g * l**2 * c * sc / D

        #   from centripetal:  m^2 l^3 theta_dot^2 sin(theta) / D
        #     = m^2 l^3 [theta_dot sin(theta)] / D  *  theta_dot (= x_4)
        A[1, 3] = m**2 * l**3 * th_d * s / D

        # theta_dd terms (row 3):
        #   from gravity:  (M+m) m g l sin(theta) / D
        #     = (M+m) m g l sinc(th) / D  *  theta  (= x_3)
        A[3, 2] = (M + m) * m * g * l * sc / D

        #   from centripetal:  -m^2 l^2 cos(th) theta_dot^2 sin(theta) / D
        #     = -m^2 l^2 cos(th) [theta_dot sin(theta)] / D  *  theta_dot
        A[3, 3] = -(m**2) * l**2 * c * th_d * s / D

        return A

    def B_continuous(self, p):
        """Continuous-time B(p). Depends only on p1 = theta.

        Parameters
        ----------
        p : (2,) array — [theta, theta_dot]

        Returns
        -------
        B : (4, 1) array
        """
        th = p[0]
        M, m, l = self.M, self.m, self.l
        D = m * l**2 * (M + m * np.sin(th) ** 2)

        B = np.zeros((4, 1))
        B[1, 0] = m * l**2 / D
        B[3, 0] = -m * l * np.cos(th) / D
        return B

    # ── Discrete-time qLPV (Euler discretization) ────────────

    def A_discrete(self, p):
        """Discrete A_d(p) = I + Ts * A_c(p)."""
        return np.eye(self.n) + self.Ts * self.A_continuous(p)

    def B_discrete(self, p):
        """Discrete B_d(p) = Ts * B_c(p)."""
        return self.Ts * self.B_continuous(p)

    # ── System tensor evaluation ─────────────────────────────

    def system_matrix(self, p, discrete=True):
        """Return S(p) = [A(p)  B(p)] at a given scheduling point.

        Parameters
        ----------
        p : (2,) array
        discrete : bool — if True, return discrete-time matrices

        Returns
        -------
        S : (n, n + n_u) array
        """
        if discrete:
            return np.hstack([self.A_discrete(p), self.B_discrete(p)])
        else:
            return np.hstack([self.A_continuous(p), self.B_continuous(p)])

    # ── Validation ───────────────────────────────────────────

    def validate_qlpv(self, n_tests=200, tol=1e-8):
        """Verify A(p)x + B(p)u = f(x,u) for random states and inputs.

        Tests the continuous-time qLPV identity. Prints results.

        Parameters
        ----------
        n_tests : int
        tol : float — max acceptable error

        Returns
        -------
        max_error : float
        """
        max_err = 0.0
        rng = np.random.default_rng(42)

        for _ in range(n_tests):
            # Random state within scheduling domain
            th = rng.uniform(-self.theta_max, self.theta_max)
            th_d = rng.uniform(-self.omega_max, self.omega_max)
            x_pos = rng.uniform(-2, 2)
            x_d = rng.uniform(-1, 1)
            x = np.array([x_pos, x_d, th, th_d])
            u = rng.uniform(-10, 10)

            # Nonlinear
            f_true = self.nonlinear_dynamics(x, u)

            # qLPV
            p = self.scheduling_param(x)
            A = self.A_continuous(p)
            B = self.B_continuous(p)
            f_qlpv = A @ x + B.flatten() * u

            err = np.max(np.abs(f_true - f_qlpv))
            max_err = max(max_err, err)

        passed = max_err < tol
        print(f"qLPV validation: max error = {max_err:.2e}  ({'PASSED' if passed else 'FAILED'})")
        return max_err

    def validate_discretization(self, n_tests=50, tol=1e-2):
        """Check that Euler discretization is close to true nonlinear step.

        Parameters
        ----------
        n_tests : int
        tol : float — max acceptable error per state component

        Returns
        -------
        max_error : float
        """
        max_err = 0.0
        rng = np.random.default_rng(123)

        for _ in range(n_tests):
            th = rng.uniform(-self.theta_max * 0.5, self.theta_max * 0.5)
            th_d = rng.uniform(-self.omega_max * 0.5, self.omega_max * 0.5)
            x = np.array([rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), th, th_d])
            u = rng.uniform(-5, 5)

            # True nonlinear step (RK4)
            x_true = self.nonlinear_step(x, u)

            # Euler qLPV step
            p = self.scheduling_param(x)
            x_euler = self.A_discrete(p) @ x + self.B_discrete(p).flatten() * u

            err = np.max(np.abs(x_true - x_euler))
            max_err = max(max_err, err)

        passed = max_err < tol
        print(
            f"Discretization validation: max error = {max_err:.2e}  "
            f"(Ts={self.Ts}, {'PASSED' if passed else 'FAILED, reduce Ts'})"
        )
        return max_err


# ── Quick test ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Inverted Pendulum on Cart — System Validation")
    print("=" * 60)

    sys = InvertedPendulum()
    print(f"\nParameters: M={sys.M}, m={sys.m}, l={sys.l}, g={sys.g}")
    print(f"State dim: {sys.n},  Input dim: {sys.n_u},  Scheduling dim: {sys.n_p}")
    print(
        f"Param bounds: theta in [{-sys.theta_max:.3f}, {sys.theta_max:.3f}], "
        f"omega in [{-sys.omega_max:.3f}, {sys.omega_max:.3f}]"
    )
    print(f"Sampling time: Ts = {sys.Ts}")

    print("\n--- Continuous-time qLPV identity ---")
    sys.validate_qlpv()

    print("\n--- Euler discretization accuracy ---")
    sys.validate_discretization()

    # Show matrices at a sample operating point
    p0 = np.array([0.3, 0.5])  # theta=0.3 rad, theta_dot=0.5 rad/s
    print(f"\n--- System matrices at p = {p0} ---")
    print(f"A_c(p) =\n{sys.A_continuous(p0)}")
    print(f"\nB_c(p) =\n{sys.B_continuous(p0).flatten()}")
    print(f"\nA_d(p) =\n{sys.A_discrete(p0)}")
    print(f"\nB_d(p) =\n{sys.B_discrete(p0).flatten()}")

    # Show that B depends on theta only (not theta_dot)
    p1 = np.array([0.3, 0.0])
    p2 = np.array([0.3, 1.5])
    b_diff = np.max(np.abs(sys.B_continuous(p1) - sys.B_continuous(p2)))
    print("\n--- B(p) independence from theta_dot ---")
    print(f"max|B(theta, 0) - B(theta, 1.5)| = {b_diff:.2e}")
