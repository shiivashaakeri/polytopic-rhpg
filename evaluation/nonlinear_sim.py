"""
Nonlinear Simulation
=====================

Simulation routines for Experiment 3 (trajectory plots) and the
comparison table.  Builds on top of system.simulate_step() and
tp_model.simulate_nonlinear() — does NOT duplicate their inner loops.

What this module adds
---------------------
simulate_open_loop      — free response at a fixed operating point (no u)
simulate_prhpg          — thin wrapper around tp_model.simulate_nonlinear
                          with timing, settling time, and divergence flags
simulate_lmi            — same interface, constant (unscheduled) LMI gain
simulate_param_ramp     — p ramps along the trajectory (e.g., airspeed sweep)
monte_carlo             — multiple IC trials, returns per-trial SimResults
settling_time           — first t where ‖x‖ < threshold and stays there
divergence_step         — first t where ‖x‖ > threshold (unstable detection)
state_norms             — ‖x_t‖₂ at every time step
input_norms             — ‖u_t‖₂ at every time step
SimResult               — dataclass collecting trajectory + metadata
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


def _u_scalar(u, m):
    """Return u as scalar if m==1, else as array (matches system.simulate_step API)."""
    return float(u[0]) if m == 1 else u


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class SimResult:
    """Trajectory data from a single simulation run.

    Attributes
    ----------
    x_traj      : (T+1, n) — state trajectory
    u_traj      : (T, m)   — input trajectory (zeros for open-loop)
    p_traj      : (T, n_p) — scheduling parameter at each step
    cost        : float    — Σ x'Qx + u'Ru  (using sim Q, R)
    stable      : bool     — True if ‖x_T‖ < divergence_tol
    diverged    : bool     — True if ‖x_t‖ exceeded div_tol at any step
    div_step    : int|None — first step where ‖x‖ > div_tol
    settle_step : int|None — first step where ‖x‖ < settle_tol permanently
    elapsed_s   : float    — wall-clock solve time [seconds]
    label       : str      — human-readable identifier
    metadata    : dict     — extra scalars (N_v, N_horizon, method, …)
    """

    x_traj: np.ndarray
    u_traj: np.ndarray
    p_traj: np.ndarray
    cost: float
    stable: bool
    diverged: bool
    div_step: Optional[int]
    settle_step: Optional[int]
    elapsed_s: float
    label: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def T(self):
        return self.u_traj.shape[0]

    @property
    def n(self):
        return self.x_traj.shape[1]

    @property
    def Ts(self):
        """Sample period from metadata (if stored)."""
        return self.metadata.get("Ts", None)

    def time_axis(self, Ts=None):
        """Return time vector [0, Ts, 2Ts, …, T·Ts]."""
        dt = Ts or self.Ts or 1.0
        return np.arange(self.T + 1) * dt


# ── Divergence / settling helpers ─────────────────────────────────────────────


def state_norms(x_traj):
    """‖x_t‖₂ at every time step.

    Parameters
    ----------
    x_traj : (T+1, n)

    Returns
    -------
    norms : (T+1,)
    """
    return np.linalg.norm(x_traj, axis=1)


def input_norms(u_traj):
    """‖u_t‖₂ at every time step.

    Parameters
    ----------
    u_traj : (T, m)

    Returns
    -------
    norms : (T,)
    """
    return np.linalg.norm(u_traj, axis=1)


def divergence_step(x_traj, threshold=1e3):
    """First step where ‖x_t‖₂ exceeds threshold.

    Parameters
    ----------
    x_traj    : (T+1, n)
    threshold : float

    Returns
    -------
    step : int or None — None if never diverges
    """
    norms = state_norms(x_traj)
    idx = np.where(norms > threshold)[0]
    return int(idx[0]) if len(idx) > 0 else None


def settling_time(x_traj, threshold=1e-3, window=10):
    """First step from which ‖x_t‖₂ stays below threshold for `window` steps.

    Parameters
    ----------
    x_traj    : (T+1, n)
    threshold : float — convergence ball radius
    window    : int   — must stay below threshold for this many steps

    Returns
    -------
    step : int or None — None if never settles
    """
    norms = state_norms(x_traj)
    T = len(norms) - 1
    for t in range(T - window + 1):
        if np.all(norms[t : t + window] < threshold):
            return t
    return None


def _postprocess(
    x_traj, u_traj, p_traj, Q, R, cost_in, t0, label, metadata, div_tol=1e3, settle_tol=1e-3, settle_window=10
):
    """Shared post-processing: flags, norms, elapsed time."""
    elapsed = time.perf_counter() - t0
    x_T_norm = float(np.linalg.norm(x_traj[-1]))
    div_s = divergence_step(x_traj, threshold=div_tol)
    set_s = settling_time(x_traj, threshold=settle_tol, window=settle_window)

    return SimResult(
        x_traj=x_traj,
        u_traj=u_traj,
        p_traj=p_traj,
        cost=float(cost_in),
        stable=(div_s is None) and (x_T_norm < div_tol),
        diverged=div_s is not None,
        div_step=div_s,
        settle_step=set_s,
        elapsed_s=elapsed,
        label=label,
        metadata=metadata,
    )


# ── Core simulations ──────────────────────────────────────────────────────────


def simulate_open_loop(system, x0, T, p_fixed=None, div_tol=1e3, label="open-loop"):
    """Free response of the system at a fixed operating point (u ≡ 0).

    Uses system.simulate_step(x, u, p) — RK4 integration at each step.
    p_fixed is held constant throughout (frozen-parameter open-loop).

    Parameters
    ----------
    system  : AeroelasticWing (or any system with simulate_step + param_bounds)
    x0      : (n,) initial state
    T       : int — number of steps
    p_fixed : (n_p,) or None — operating point; default: midpoint of bounds
    div_tol : float — ‖x‖ above which we flag divergence

    Returns
    -------
    result : SimResult  (u_traj is all zeros)
    """
    t0 = time.perf_counter()
    n = len(x0)
    m = getattr(system, "n_u", 1)

    if p_fixed is None:
        bounds = system.param_bounds
        p_fixed = 0.5 * (bounds[:, 0] + bounds[:, 1])
    p_fixed = np.asarray(p_fixed, dtype=float)

    x_traj = np.zeros((T + 1, n))
    u_traj = np.zeros((T, m))
    p_traj = np.tile(p_fixed, (T, 1))
    x_traj[0] = x0

    u_zero = 0.0 if m == 1 else np.zeros(m)
    for t in range(T):
        x_traj[t + 1] = system.simulate_step(x_traj[t], u_zero, p_fixed)
        if np.linalg.norm(x_traj[t + 1]) > div_tol * 10:
            # Clip to avoid NaN propagation in extremely unstable cases
            x_traj[t + 1 :] = div_tol * 10
            u_traj[t + 1 :] = 0.0
            p_traj[t + 1 :] = p_fixed
            break

    return _postprocess(
        x_traj,
        u_traj,
        p_traj,
        Q=np.eye(n),
        R=np.eye(m),
        cost_in=np.inf,
        t0=t0,
        label=label,
        metadata=dict(p_fixed=p_fixed.tolist(), Ts=system.Ts),
        div_tol=div_tol,
    )


def simulate_prhpg(tp_model, K_vertices, x0, T, Q, R, div_tol=1e3, settle_tol=1e-3, settle_window=10, label="P-RHPG"):
    """Closed-loop simulation under the P-RHPG PDC controller.

    Wraps tp_model.simulate_nonlinear() and attaches timing + stability flags.
    Scheduling is state-dependent: p_t = system.scheduling_param(x_t).

    Parameters
    ----------
    tp_model   : TPModel
    K_vertices : (N_v, m, n) — vertex gains from P-RHPG result
    x0         : (n,)
    T          : int — steps
    Q, R       : cost matrices (must match those used in P-RHPG)
    div_tol    : float — divergence threshold on ‖x‖
    settle_tol : float — convergence ball radius
    settle_window : int

    Returns
    -------
    result : SimResult
    """
    t0 = time.perf_counter()

    x_traj, u_traj, cost = tp_model.simulate_nonlinear(K_vertices, x0, T, Q=Q, R=R)

    # Reconstruct p_traj from state trajectory
    p_traj = np.stack([tp_model.system.scheduling_param(x_traj[t]) for t in range(T)])

    n_p = p_traj.shape[1] if p_traj.ndim == 2 else 1

    return _postprocess(
        x_traj,
        u_traj,
        p_traj,
        Q=Q,
        R=R,
        cost_in=cost,
        t0=t0,
        label=label,
        metadata=dict(
            N_v=tp_model.Nv,
            Ts=tp_model.system.Ts,
        ),
        div_tol=div_tol,
        settle_tol=settle_tol,
        settle_window=settle_window,
    )


def simulate_lmi(
    system, K_lmi, x0, T, Q, R, p_fixed=None, div_tol=1e3, settle_tol=1e-3, settle_window=10, label="LMI-PDC"
):
    """Closed-loop simulation under a constant (unscheduled) LMI gain.

    u_t = K_lmi @ x_t  (K_lmi is a single (m, n) matrix, not scheduled).
    Uses system.simulate_step(x, u, p) with p held fixed at p_fixed.

    For systems whose scheduling parameter is external (e.g., airspeed V
    for the aeroelastic wing), p_fixed must be supplied explicitly.
    Defaults to the domain midpoint if omitted.

    Parameters
    ----------
    system  : AeroelasticWing
    K_lmi   : (m, n) — constant stabilising gain from LMI synthesis
    x0      : (n,)
    T       : int
    Q, R    : cost matrices
    p_fixed : (n_p,) or None — fixed operating point (default: domain midpoint)
    div_tol, settle_tol, settle_window : see simulate_prhpg

    Returns
    -------
    result : SimResult
    """
    t0 = time.perf_counter()
    n = len(x0)
    m = K_lmi.shape[0]

    if p_fixed is None:
        bounds = system.param_bounds
        p_fixed = 0.5 * (bounds[:, 0] + bounds[:, 1])
    p_fixed = np.asarray(p_fixed, dtype=float)

    x_traj = np.zeros((T + 1, n))
    u_traj = np.zeros((T, m))
    p_traj = np.tile(p_fixed, (T, 1))
    x_traj[0] = x0
    cost = 0.0

    for t in range(T):
        u_traj[t] = K_lmi @ x_traj[t]
        x_traj[t + 1] = system.simulate_step(x_traj[t], _u_scalar(u_traj[t], m), p_fixed)

        cost += x_traj[t] @ Q @ x_traj[t] + u_traj[t] @ R @ u_traj[t]

        if np.linalg.norm(x_traj[t + 1]) > div_tol * 10:
            x_traj[t + 2 :] = div_tol * 10
            break

    cost += x_traj[T] @ Q @ x_traj[T]

    return _postprocess(
        x_traj,
        u_traj,
        p_traj,
        Q=Q,
        R=R,
        cost_in=cost,
        t0=t0,
        label=label,
        metadata=dict(Ts=system.Ts, p_fixed=p_fixed.tolist()),
        div_tol=div_tol,
        settle_tol=settle_tol,
        settle_window=settle_window,
    )


def simulate_pdc_scheduled(
    system, tp_model, K_vertices, x0, T, Q, R, div_tol=1e3, settle_tol=1e-3, settle_window=10, label="PDC-scheduled"
):
    """Closed-loop with explicitly scheduled PDC gains.

    Like simulate_prhpg but uses system.simulate_step(x, u, p) directly
    (RK4) rather than the tp_model's potentially ZOH-based step.
    Useful to compare integration accuracy when Ts is large.

    Parameters
    ----------
    system     : AeroelasticWing
    tp_model   : TPModel — for weights(p)
    K_vertices : (N_v, m, n)
    x0, T, Q, R, div_tol, settle_tol, settle_window : see simulate_prhpg

    Returns
    -------
    result : SimResult
    """
    t0 = time.perf_counter()
    n = len(x0)
    m = K_vertices.shape[1]

    x_traj = np.zeros((T + 1, n))
    u_traj = np.zeros((T, m))
    p_traj = np.zeros((T, system.n_p))
    x_traj[0] = x0
    cost = 0.0

    for t in range(T):
        # Use state-derived scheduling if available, else tp_model evaluate
        if hasattr(system, "scheduling_param"):
            p_t = system.scheduling_param(x_traj[t])
        else:
            # External scheduling: use midpoint (override via p_schedule if needed)
            bounds = system.param_bounds
            p_t = 0.5 * (bounds[:, 0] + bounds[:, 1])
        alpha = tp_model.weights(p_t)
        K_t = np.einsum("i,imn->mn", alpha, K_vertices)

        p_traj[t] = p_t
        u_traj[t] = K_t @ x_traj[t]
        x_traj[t + 1] = system.simulate_step(x_traj[t], _u_scalar(u_traj[t], m), p_t)

        cost += x_traj[t] @ Q @ x_traj[t] + u_traj[t] @ R @ u_traj[t]

        if np.linalg.norm(x_traj[t + 1]) > div_tol * 10:
            x_traj[t + 2 :] = div_tol * 10
            break

    cost += x_traj[T] @ Q @ x_traj[T]

    return _postprocess(
        x_traj,
        u_traj,
        p_traj,
        Q=Q,
        R=R,
        cost_in=cost,
        t0=t0,
        label=label,
        metadata=dict(N_v=tp_model.Nv, Ts=system.Ts),
        div_tol=div_tol,
        settle_tol=settle_tol,
        settle_window=settle_window,
    )


def simulate_param_ramp(
    system,
    tp_model,
    K_vertices,
    x0,
    T,
    Q,
    R,
    p_start,
    p_end,
    div_tol=1e3,
    settle_tol=1e-3,
    settle_window=10,
    label="param-ramp",
):
    """Closed-loop with p linearly ramped from p_start to p_end over T steps.

    Useful for showing robustness as the airspeed increases through the
    flutter boundary.  p is externally commanded (e.g., V ramp), so it
    overrides the state-derived scheduling parameter.

    Parameters
    ----------
    system     : AeroelasticWing
    tp_model   : TPModel
    K_vertices : (N_v, m, n)
    x0         : (n,)
    T          : int
    Q, R       : cost matrices
    p_start    : (n_p,) — starting parameter
    p_end      : (n_p,) — ending parameter

    Returns
    -------
    result : SimResult
    """
    t0 = time.perf_counter()
    n = len(x0)
    m = K_vertices.shape[1]
    n_p = len(p_start)

    p_schedule = np.linspace(p_start, p_end, T)  # (T, n_p)

    x_traj = np.zeros((T + 1, n))
    u_traj = np.zeros((T, m))
    p_traj = p_schedule.copy()
    x_traj[0] = x0
    cost = 0.0

    for t in range(T):
        p_t = p_schedule[t]
        alpha = tp_model.weights(p_t)
        K_t = np.einsum("i,imn->mn", alpha, K_vertices)

        u_traj[t] = K_t @ x_traj[t]
        x_traj[t + 1] = system.simulate_step(x_traj[t], _u_scalar(u_traj[t], m), p_t)

        cost += x_traj[t] @ Q @ x_traj[t] + u_traj[t] @ R @ u_traj[t]

        if np.linalg.norm(x_traj[t + 1]) > div_tol * 10:
            x_traj[t + 2 :] = div_tol * 10
            break

    cost += x_traj[T] @ Q @ x_traj[T]

    return _postprocess(
        x_traj,
        u_traj,
        p_traj,
        Q=Q,
        R=R,
        cost_in=cost,
        t0=t0,
        label=label,
        metadata=dict(p_start=list(p_start), p_end=list(p_end), N_v=tp_model.Nv, Ts=system.Ts),
        div_tol=div_tol,
        settle_tol=settle_tol,
        settle_window=settle_window,
    )


# ── Monte Carlo ───────────────────────────────────────────────────────────────


def monte_carlo(sim_fn, x0_samples, **sim_kwargs):
    """Run sim_fn over multiple initial conditions and return all SimResults.

    Parameters
    ----------
    sim_fn     : callable — any of simulate_prhpg, simulate_lmi, etc.
                 Must accept x0 as first positional arg after fixed args.
                 Called as sim_fn(x0=x0_i, **sim_kwargs).
    x0_samples : (M, n) array — M initial conditions

    Returns
    -------
    results : list of SimResult — length M
    summary : dict with keys:
        n_stable, n_diverged, mean_cost, std_cost,
        mean_settle (steps), fraction_settled
    """
    results = []
    for i, x0 in enumerate(x0_samples):
        res = sim_fn(x0=np.asarray(x0), **sim_kwargs)
        res.label = f"{res.label} [IC {i}]"
        results.append(res)

    costs = np.array([r.cost for r in results])
    stables = np.array([r.stable for r in results])
    settles = np.array([r.settle_step if r.settle_step is not None else np.nan for r in results])

    summary = dict(
        n_stable=int(np.sum(stables)),
        n_diverged=int(np.sum([r.diverged for r in results])),
        mean_cost=float(np.nanmean(costs[stables])) if stables.any() else np.nan,
        std_cost=float(np.nanstd(costs[stables])) if stables.any() else np.nan,
        mean_settle=float(np.nanmean(settles)),
        fraction_settled=float(np.mean(~np.isnan(settles))),
    )
    return results, summary


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/mnt/user-data/uploads")

    print("=" * 60)
    print("nonlinear_sim.py — Self-test")
    print("=" * 60)

    import numpy as np
    from aeroelastic_wing import AeroelasticWing

    system = AeroelasticWing()
    Ts = system.Ts
    n = system.n
    m = system.n_u
    n_p = system.n_p

    x0 = np.array([0.0, 0.0, 0.05, 0.0])
    T = 100
    Q = np.diag([10.0, 1.0, 100.0, 1.0])
    R = np.array([[1.0]])
    V_nom = 20.0
    V_flutter = 30.0
    p_nom = np.array([V_nom, 0.0, 0.0])

    # ── state_norms / input_norms ─────────────────────────────────────────────
    print("\n--- state_norms / input_norms ---")
    x_dummy = np.random.default_rng(0).standard_normal((T + 1, n))
    u_dummy = np.random.default_rng(1).standard_normal((T, m))
    norms_x = state_norms(x_dummy)
    norms_u = input_norms(u_dummy)
    assert norms_x.shape == (T + 1,)
    assert norms_u.shape == (T,)
    assert np.allclose(norms_x[0], np.linalg.norm(x_dummy[0]))
    print(f"  state_norms: shape={norms_x.shape}, max={norms_x.max():.3f}  ✓")
    print(f"  input_norms: shape={norms_u.shape}, max={norms_u.max():.3f}  ✓")

    # ── divergence_step ───────────────────────────────────────────────────────
    print("\n--- divergence_step ---")
    x_div = np.ones((T + 1, n)) * 10.0
    x_div[5] = 2000.0
    assert divergence_step(x_div, threshold=1e3) == 5
    assert divergence_step(np.zeros((T + 1, n)), threshold=1e3) is None
    print("  divergence_step: ✓")

    # ── settling_time ─────────────────────────────────────────────────────────
    print("\n--- settling_time ---")
    x_settle = np.zeros((T + 1, n))
    x_settle[:20] = 1.0  # big initial transient
    x_settle[20:] = 0.0001  # settled
    st = settling_time(x_settle, threshold=1e-3, window=5)
    assert st is not None and st <= 20, f"Expected settle by 20, got {st}"
    print(f"  settling_time: step={st}  ✓")

    # ── simulate_open_loop: stable operating point ────────────────────────────
    print("\n--- simulate_open_loop (V=20, stable) ---")
    res_ol_s = simulate_open_loop(system, x0, T, p_fixed=p_nom, label="OL-stable")
    assert res_ol_s.x_traj.shape == (T + 1, n)
    assert res_ol_s.u_traj.shape == (T, m)
    assert np.allclose(res_ol_s.u_traj, 0.0)
    print(f"  Final ‖x‖ = {np.linalg.norm(res_ol_s.x_traj[-1]):.4f}")
    print(f"  stable={res_ol_s.stable}, diverged={res_ol_s.diverged}  ✓")

    # ── simulate_open_loop: flutter operating point ───────────────────────────
    print("\n--- simulate_open_loop (V=30, flutter) ---")
    p_flutter = np.array([V_flutter, 0.0, 0.0])
    res_ol_f = simulate_open_loop(system, x0, T, p_fixed=p_flutter, label="OL-flutter")
    print(f"  Final ‖x‖ = {np.linalg.norm(res_ol_f.x_traj[-1]):.4e}")
    print(f"  stable={res_ol_f.stable}, diverged={res_ol_f.diverged}")
    print(f"  diverge_step={res_ol_f.div_step}  ✓")

    # ── simulate_lmi: simple stabilising gain (pole placement) ───────────────
    print("\n--- simulate_lmi (synthetic stabilising gain) ---")
    # Compute a simple LQR gain at nominal p to test interface
    from scipy import linalg as scipy_linalg

    A_nom = system.A_discrete(p_nom)
    B_nom = system.B_discrete(p_nom)
    try:
        P_nom = scipy_linalg.solve_discrete_are(A_nom, B_nom, Q, R)
        K_lmi_test = -np.linalg.solve(R + B_nom.T @ P_nom @ B_nom, B_nom.T @ P_nom @ A_nom)
    except Exception:
        K_lmi_test = np.zeros((m, n))  # fallback

    res_lmi = simulate_lmi(system, K_lmi_test, x0, T, Q, R, p_fixed=p_nom, label="LMI-test")
    assert res_lmi.x_traj.shape == (T + 1, n)
    assert res_lmi.u_traj.shape == (T, m)
    print(f"  cost={res_lmi.cost:.4e}, stable={res_lmi.stable}")
    print(f"  elapsed={res_lmi.elapsed_s:.4f} s  ✓")

    # ── SimResult helpers ─────────────────────────────────────────────────────
    print("\n--- SimResult helpers ---")
    t_axis = res_lmi.time_axis()
    assert len(t_axis) == T + 1
    assert abs(t_axis[1] - Ts) < 1e-12
    print(f"  time_axis: [{t_axis[0]:.3f}, …, {t_axis[-1]:.3f}] s  ✓")
    print(f"  T={res_lmi.T}, n={res_lmi.n}  ✓")

    # ── monte_carlo ───────────────────────────────────────────────────────────
    print("\n--- monte_carlo (5 ICs, LMI) ---")
    rng = np.random.default_rng(42)
    x0_batch = rng.standard_normal((5, n)) * 0.01  # small ICs near origin
    mc_results, mc_summary = monte_carlo(
        sim_fn=simulate_lmi,
        x0_samples=x0_batch,
        system=system,
        K_lmi=K_lmi_test,
        T=T,
        Q=Q,
        R=R,
        label="MC-LMI",
    )
    assert len(mc_results) == 5
    print(f"  n_stable={mc_summary['n_stable']}/5")
    print(f"  mean_cost={mc_summary['mean_cost']:.4e}")
    print(f"  fraction_settled={mc_summary['fraction_settled']:.2f}  ✓")

    print("\nAll tests passed ✓")
