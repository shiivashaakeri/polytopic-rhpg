"""
Comparison Plots
================

Publication-quality figures for the CDC 2026 paper.

Figure 1 — Cross-term stability (Experiment 1)
    Scatter: ρ(A_i) open-loop  vs  ρ(G_{ii}) vertex CL  vs  ρ(A_cl(p; K̄))
    joint CL over a fine grid.  Motivates the PDC cross-term problem.

Figure 2 — Convergence in N (Experiment 2)
    J_∞(K̄_N) vs horizon N, with Riccati lower bound J_ric and fitted
    geometric decay rate.  Validates Theorem 4.1 (monotone, bounded, convergent).

Figure 3 — Closed-loop trajectory (Experiment 3)
    State trajectories: open-loop (diverging) vs P-RHPG (stabilised).
    Shows pitch angle α(t) and plunge h(t) for the aeroelastic wing.

Table — P-RHPG vs LMI comparison
    Columns: method, N_v, J_∞, Tr(W) (LMI bound), solve time, feasible.
    Saved as both a formatted CSV and a LaTeX tabular fragment.

All functions write to FIG_DIR from config.py and return the figure object.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path
import csv, io

# ── Style ─────────────────────────────────────────────────────────────────────

IEEE_HALF  = 3.5    # single column [in]
IEEE_FULL  = 7.16   # double column [in]
IEEE_H     = 2.4    # typical height [in]
DPI        = 300

COLORS = {
    'open'   : '#d62728',   # red    — open-loop / unstable
    'vertex' : '#ff7f0e',   # orange — vertex-only CL
    'joint'  : '#1f77b4',   # blue   — joint CL (P-RHPG)
    'lmi'    : '#2ca02c',   # green  — LMI-PDC
    'riccati': '#9467bd',   # purple — Riccati lower bound
    'stable' : '#1f77b4',
    'unstable': '#d62728',
}
MARKERS = {'open': 's', 'vertex': '^', 'joint': 'o'}


def _ieee_fig(w=IEEE_HALF, h=IEEE_H):
    """Create a figure pre-configured for IEEE two-column format."""
    plt.rcParams.update({
        'font.family'     : 'serif',
        'font.size'       : 8,
        'axes.labelsize'  : 8,
        'axes.titlesize'  : 8,
        'xtick.labelsize' : 7,
        'ytick.labelsize' : 7,
        'legend.fontsize' : 7,
        'lines.linewidth' : 1.2,
        'lines.markersize': 4,
        'axes.grid'       : True,
        'grid.alpha'      : 0.3,
        'grid.linestyle'  : '--',
        'figure.dpi'      : DPI,
    })
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def _save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {path}')
    return path


# ── Figure 1: Cross-term stability ───────────────────────────────────────────

def plot_cross_term_stability(stab_dict, save_path=None, title=None):
    """Figure 1 — spectral radius scatter plot.

    Shows three distributions:
      (a) ρ(A_i) open-loop vertex matrices    — should be > 1 (unstable)
      (b) ρ(G_{ii}) vertex closed-loop        — all < 1 (vertex stable)
      (c) ρ(A_cl(p; K̄)) joint closed-loop    — all < 1 (jointly stable)

    The gap between (b) and (c) is the cross-term effect: vertex stability
    does not imply joint stability.

    Parameters
    ----------
    stab_dict : dict from joint_stability.stability_comparison()
    save_path : Path or str or None
    title     : str or None

    Returns
    -------
    fig, ax
    """
    fig, ax = _ieee_fig(w=IEEE_HALF, h=IEEE_H)

    rho_open   = stab_dict['rho_open']      # (N_v,)
    rho_diag   = stab_dict['rho_diagonal']  # (N_v,)
    rho_joint  = stab_dict['rho_joint']     # (N_eval,)

    N_v    = len(rho_open)
    N_eval = len(rho_joint)

    # Jitter x-positions so overlapping points are visible
    rng = np.random.default_rng(0)
    x_open   = np.ones(N_v)    * 1 + rng.uniform(-0.08, 0.08, N_v)
    x_vertex = np.ones(N_v)    * 2 + rng.uniform(-0.08, 0.08, N_v)
    x_joint  = np.ones(N_eval) * 3 + rng.uniform(-0.10, 0.10, N_eval)

    ax.scatter(x_open,   rho_open,  color=COLORS['open'],
               marker=MARKERS['open'],   label=r'$\rho(A_i)$ open-loop',
               alpha=0.85, zorder=3)
    ax.scatter(x_vertex, rho_diag,  color=COLORS['vertex'],
               marker=MARKERS['vertex'], label=r'$\rho(G_{ii})$ vertex CL',
               alpha=0.85, zorder=3)
    ax.scatter(x_joint,  rho_joint, color=COLORS['joint'],
               marker=MARKERS['joint'],  label=r'$\rho(A_\mathrm{cl}(p;\bar{K}))$ joint CL',
               alpha=0.6,  zorder=3, s=10)

    # Stability boundary
    ax.axhline(1.0, color='k', lw=1.0, ls='--', label=r'$\rho = 1$')

    # Shade unstable region
    ymax = max(np.max(rho_open) * 1.05, 1.1)
    ax.axhspan(1.0, ymax, alpha=0.06, color=COLORS['open'], zorder=0)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Open-loop', 'Vertex CL', 'Joint CL'], fontsize=7)
    ax.set_xlim(0.5, 3.7)
    ax.set_ylim(bottom=max(0, np.min(rho_joint) * 0.95), top=ymax)
    ax.set_ylabel(r'Spectral radius $\rho$')
    ax.legend(loc='upper right', framealpha=0.9)

    if title:
        ax.set_title(title, pad=3)

    # Annotate counts
    ax.text(1, ax.get_ylim()[0] + 0.02 * (ymax - ax.get_ylim()[0]),
            f'$N_v={N_v}$', ha='center', va='bottom', fontsize=6, color='gray')
    ax.text(3, ax.get_ylim()[0] + 0.02 * (ymax - ax.get_ylim()[0]),
            f'$N_q={N_eval}$', ha='center', va='bottom', fontsize=6, color='gray')

    if save_path:
        _save(fig, save_path)
    return fig, ax


# ── Figure 2: Convergence in N ────────────────────────────────────────────────
def plot_convergence(Ns, J_Ns, J_bar=None, J_feas=None,
                     save_path=None, title=None,
                     # kept for backward-compat but ignored:
                     J_infs=None, J_ric=None, rate=None, C=None,
                     conv_idx=None):
    """Figure 2 — Validates Theorem 2 (all three parts).

    Shows only quantities that appear in the theorem statement:

      J_N*   (Thm 2-i)  : monotonically non-decreasing in N
      J_bar  (Thm 2-iii): horizontal limit line  J_N* -> J_bar
      J_feas (Thm 2-ii) : horizontal upper-bound  J_N* <= J_inf(K^feas)

    Everything else (J_inf(K_bar_N), J_ric, fit curve) is intentionally omitted.

    Parameters
    ----------
    Ns     : (K,) array-like -- horizon values (x-axis)
    J_Ns   : (K,) array-like -- finite-horizon optimal cost J_N* (monotone up)
    J_bar  : float or None   -- limit J_bar approx J_N*(N=1000), dotted line
    J_feas : float or None   -- J_inf(K^feas) upper bound, dashed line
    save_path, title : optional
    """
    fig, ax = _ieee_fig(w=IEEE_HALF, h=IEEE_H)

    Ns   = np.asarray(Ns)
    J_Ns = np.asarray(J_Ns)

    # -- J_N* -- the only curve -----------------------------------------------
    valid = np.isfinite(J_Ns)
    ax.semilogy(Ns[valid], J_Ns[valid],
                color=COLORS['joint'], marker='s', ms=4, lw=1.4,
                label=r'$J_N^*$ (finite-horizon optimal)', zorder=4)

    # -- J_bar: limit line (Thm 2-iii) ----------------------------------------
    if J_bar is not None and np.isfinite(J_bar):
        ax.axhline(J_bar, color='k', lw=1.0, ls=':',
                   label=r'$\bar{J}$ (limit, $N\!=\!1000$)', zorder=3)

    # -- J_inf(K^feas): upper bound (Thm 2-ii) --------------------------------
    if J_feas is not None and np.isfinite(J_feas):
        ax.axhline(J_feas, color=COLORS['riccati'], lw=1.0, ls='--',
                   label=r'$J_\infty(\mathbf{K}^{\mathrm{feas}})$ (Thm.~2-ii)',
                   zorder=3)

    # -- y-axis limits --------------------------------------------------------
    y_lo = J_Ns[valid].min() * 0.7
    y_hi = (J_feas if (J_feas and np.isfinite(J_feas))
            else J_Ns[valid].max()) * 1.15
    ax.set_ylim(y_lo, y_hi)

    # -- Axes labels & legend -------------------------------------------------
    ax.set_xlabel(r'Horizon $N$')
    ax.set_ylabel(r'$J_N^*$')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax.legend(loc='lower right', framealpha=0.9)

    if title:
        ax.set_title(title, pad=3)

    if save_path:
        _save(fig, save_path)
    return fig, ax

# ── Figure 3: Closed-loop trajectory ─────────────────────────────────────────

def plot_trajectory(res_open, res_prhpg,
                    state_indices=(0, 2),
                    state_labels=(r'Plunge $h$ [m]', r'Pitch $\alpha$ [rad]'),
                    res_lmi=None,
                    save_path=None, title=None,
                    zoom_window=(0.5, 1.5),
                    res_extra=None):
    """Figure 3: P-RHPG vs LMI-PDC closed-loop trajectories.

    Open-loop is omitted; a vertical dashed line marks its divergence time.
    The pitch subplot carries a zoomed inset over zoom_window = (t0, t1).

    Parameters
    ----------
    zoom_window : (float, float)
        Time interval [t0, t1] shown in the pitch inset.
    res_extra : list of (SimResult, label, color, linestyle), optional
        Additional baselines (e.g. a nominal LQR gain) drawn in main + inset.
    """
    res_extra = res_extra or []
    n_states = len(state_indices)
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7,
        'lines.linewidth': 1.2, 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.linestyle': '--',
    })
    fig, axes = plt.subplots(n_states, 1,
                              figsize=(IEEE_HALF, IEEE_H * n_states * 0.85),
                              sharex=True)
    if n_states == 1:
        axes = [axes]

    Ts      = res_prhpg.metadata.get('Ts', 0.002)
    t_prhpg = res_prhpg.time_axis(Ts)
    T_total = t_prhpg[-1]

    div_step = res_open.div_step if res_open is not None else None
    div_time = div_step * Ts if div_step is not None else None

    if res_lmi is not None:
        t_lmi = res_lmi.time_axis(Ts)

    for k, (idx, label) in enumerate(zip(state_indices, state_labels)):
        ax = axes[k]

        # ── Main plot ─────────────────────────────────────────────────────────
        ax.plot(t_prhpg, res_prhpg.x_traj[:, idx],
                color=COLORS['joint'], lw=1.2, label='P-RHPG', zorder=3)
        if res_lmi is not None:
            ax.plot(t_lmi, res_lmi.x_traj[:, idx],
                    color=COLORS['lmi'], lw=1.0, ls='-.', label='LMI-PDC',
                    zorder=3)
        for (r_e, lab_e, col_e, ls_e) in res_extra:
            ax.plot(r_e.time_axis(Ts), r_e.x_traj[:, idx],
                    color=col_e, lw=1.0, ls=ls_e, label=lab_e, zorder=2)

        # ── y-limits from full window ─────────────────────────────────────────
        # y-limits from P-RHPG and the extra baselines only; a slowly
        # diverging LMI curve is clipped rather than allowed to flatten the rest
        vals = list(res_prhpg.x_traj[:, idx])
        for (r_e, *_ ) in res_extra:
            vals += list(r_e.x_traj[:, idx])
        v_max = max(np.max(np.abs(vals)), 1e-4)
        pad   = v_max * 0.25
        ax.set_ylim(-(v_max + pad), v_max + pad)

        # ── Divergence marker ─────────────────────────────────────────────────
        if div_time is not None:
            ax.axvline(div_time, color=COLORS['open'], lw=0.9,
                       ls='--', alpha=0.6, zorder=1)
            if k == 0:
                ylo, yhi = ax.get_ylim()
                ax.text(div_time - T_total * 0.015, yhi * 0.88,
                        f'Open-loop\ndiverges\n$t={div_time:.1f}$ s',
                        fontsize=6, color=COLORS['open'], va='top',
                        ha='right', linespacing=1.3)

        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_ylabel(label)

        if k == 0:
            ax.legend(loc='lower right', framealpha=0.9)

        # ── Zoomed inset on the pitch (last) subplot ──────────────────────────
        if k == n_states - 1:
            # Inset in top-right corner, clear of the legend
            axin = ax.inset_axes([0.52, 0.38, 0.46, 0.58])

            axin.plot(t_prhpg, res_prhpg.x_traj[:, idx],
                      color=COLORS['joint'], lw=1.0, zorder=3)
            if res_lmi is not None:
                axin.plot(t_lmi, res_lmi.x_traj[:, idx],
                          color=COLORS['lmi'], lw=0.9, ls='-.', zorder=3)
            for (r_e, lab_e, col_e, ls_e) in res_extra:
                axin.plot(r_e.time_axis(Ts), r_e.x_traj[:, idx],
                          color=col_e, lw=0.9, ls=ls_e, zorder=2)
            axin.axhline(0, color='k', lw=0.4, ls=':')

            # y-limits from the zoom window only
            z0, z1 = zoom_window
            mask_z = (t_prhpg >= z0) & (t_prhpg <= z1)
            zvals  = list(res_prhpg.x_traj[mask_z, idx])
            if res_lmi is not None:
                mask_zl = (t_lmi >= z0) & (t_lmi <= z1)
                zvals  += list(res_lmi.x_traj[mask_zl, idx])
            for (r_e, *_ ) in res_extra:
                t_e = r_e.time_axis(Ts); mask_e = (t_e >= z0) & (t_e <= z1)
                zvals += list(r_e.x_traj[mask_e, idx])
            vz_max = max(np.max(np.abs(zvals)), 1e-4)
            zpad   = vz_max * 0.20
            axin.set_xlim(z0, z1)
            axin.set_ylim(-(vz_max + zpad), vz_max + zpad)
            axin.tick_params(labelsize=5.5)
            axin.set_xlabel(r'$t$ [s]', fontsize=6)
            axin.grid(True, alpha=0.25, linestyle='--')
            axin.set_facecolor('white')

            # Draw zoom indicator: box on main axes + two connector lines
            ax.indicate_inset_zoom(axin, edgecolor='black', alpha=0.5, lw=0.7)

    axes[-1].set_xlabel(r'Time [s]')
    axes[-1].set_xlim(0, T_total)

    if title:
        axes[0].set_title(title, pad=3)

    if save_path:
        fig.tight_layout()
        _save(fig, save_path, tight=False)
    return fig, axes



# ── Table: P-RHPG vs LMI comparison ──────────────────────────────────────────

def format_comparison_table(rows, save_path_csv=None, save_path_tex=None):
    """Format and optionally save the comparison table.

    Parameters
    ----------
    rows : list of dict, each with keys:
        method      : str  ('P-RHPG' or 'LMI-PDC')
        N_v         : int
        J_inf       : float — integrated infinite-horizon cost
        obj_bound   : float — Tr(W) for LMI, J_N for P-RHPG (or NaN)
        J_ric       : float — Riccati lower bound
        gap_pct     : float — (J_inf - J_ric) / J_ric * 100
        solve_time  : float — [s]
        feasible    : bool
        stable      : bool

    Returns
    -------
    table_str : str — formatted plain-text table
    latex_str : str — LaTeX tabular fragment
    """
    cols = ['Method', r'$N_v$', r'$J_\infty$', 'Bound',
            r'$J_\mathrm{ric}$', 'Gap [%]', 'Time [s]', 'Stable']

    # Plain text
    lines = []
    hdr   = f"{'Method':<12} {'Nv':>4} {'J_inf':>12} {'Bound':>12} "
    hdr  += f"{'J_ric':>12} {'Gap%':>8} {'Time':>8} {'Stable':>6}"
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for r in rows:
        J_inf = r.get('J_inf', float('nan'))
        bound = r.get('obj_bound', float('nan'))
        J_ric = r.get('J_ric', float('nan'))
        gap   = r.get('gap_pct', float('nan'))
        t     = r.get('solve_time', float('nan'))
        line  = (f"{r['method']:<12} {r['N_v']:>4} "
                 f"{J_inf:>12.4e} {bound:>12.4e} "
                 f"{J_ric:>12.4e} {gap:>8.2f} "
                 f"{t:>8.3f} {'✓' if r.get('stable', False) else '✗':>6}")
        lines.append(line)
    table_str = '\n'.join(lines)

    # LaTeX
    # Group rows by N_v; within each group P-RHPG comes before LMI-PDC
    # Columns: Method | N_v | J_inf | Gap[%] | Time[s] | Stable
    # (Bound column dropped — J_N and Tr(W) are incommensurable)
    tex_lines = [
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Method & $N_v$ & $J_\infty$ & Gap [\%] & Time [s] & Stable \\',
        r'\midrule',
    ]
    prev_nv = None
    for r in rows:
        nv = r['N_v']
        # Horizontal rule between N_v groups
        if prev_nv is not None and nv != prev_nv:
            tex_lines.append(r'\midrule')
        prev_nv = nv

        J_inf = r.get('J_inf', float('nan'))
        gap   = r.get('gap_pct', float('nan'))
        t     = r.get('solve_time', float('nan'))
        stable_sym = r'\checkmark' if r.get('stable', False) else r'\texttimes'

        def fmt(v, prec=2, sci=True):
            if not np.isfinite(v):
                return '--'
            if sci:
                return f'${v:.{prec}e}$'.replace('e+0', r'{\times}10^{').replace('e-0', r'{\times}10^{-').replace('e+', r'{\times}10^{').replace('e-', r'{\times}10^{-') + '}'
            return f'{v:.{prec}f}'

        # Simpler formatting for the paper
        J_str  = f'${J_inf:.2e}$' if np.isfinite(J_inf) else r'\textemdash'
        gap_str = f'{gap:.1f}' if np.isfinite(gap) else r'\textemdash'

        tex_lines.append(
            fr'{r["method"]} & {nv} & {J_str} '
            fr'& {gap_str} & {t:.1f} & {stable_sym} \\'
        )
    tex_lines += [r'\bottomrule', r'\end{tabular}']
    latex_str = '\n'.join(tex_lines)

    if save_path_csv is not None:
        save_path_csv = Path(save_path_csv)
        with open(save_path_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'method', 'N_v', 'J_inf', 'obj_bound', 'J_ric',
                'gap_pct', 'solve_time', 'feasible', 'stable'])
            writer.writeheader()
            writer.writerows(rows)
        print(f'  Saved CSV → {save_path_csv}')

    if save_path_tex is not None:
        save_path_tex = Path(save_path_tex)
        save_path_tex.write_text(latex_str)
        print(f'  Saved LaTeX → {save_path_tex}')

    return table_str, latex_str


# ── Supplementary: cost surface 1D slice ─────────────────────────────────────

def plot_cost_slice_1d(p_vals, pdc_costs, ric_costs,
                       rho_vals=None,
                       xlabel='Airspeed $V$ [m/s]',
                       flutter_speed=None,
                       save_path=None, title=None):
    """Plot J_∞(p; K) and J_ric(p) as functions of one scheduling variable.

    Parameters
    ----------
    p_vals     : (K,) — values along the varied dimension
    pdc_costs  : (K,) — J_∞(p; K) from frozen_param_cost.marginal_cost_1d
    ric_costs  : (K,) — J_ric(p) from riccati_cost_grid (or None)
    rho_vals   : (K,) or None — ρ(A_cl(p)) on twin y-axis
    xlabel     : str
    flutter_speed : float or None — vertical line marking flutter boundary
    save_path, title : optional

    Returns
    -------
    fig, ax (or (fig, ax, ax2) if rho_vals given)
    """
    fig, ax = _ieee_fig(w=IEEE_HALF, h=IEEE_H)

    valid_pdc = np.isfinite(pdc_costs)
    ax.plot(p_vals[valid_pdc], pdc_costs[valid_pdc],
            color=COLORS['joint'], label=r'$J_\infty(p;\bar{K})$ P-RHPG')

    if ric_costs is not None:
        valid_ric = np.isfinite(ric_costs)
        ax.plot(p_vals[valid_ric], ric_costs[valid_ric],
                color=COLORS['riccati'], ls=':', label=r'$J_\mathrm{ric}(p)$')

    if flutter_speed is not None:
        ax.axvline(flutter_speed, color=COLORS['open'], lw=0.9, ls='--',
                   label=fr'Flutter $V_f={flutter_speed:.1f}$ m/s')
        ax.axvspan(flutter_speed, p_vals.max(), alpha=0.05, color=COLORS['open'])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Cost $J_\infty$')
    ax.legend(loc='upper left', framealpha=0.9)

    ax2 = None
    if rho_vals is not None:
        ax2 = ax.twinx()
        ax2.plot(p_vals, rho_vals, color='gray', lw=0.8, ls=':',
                 label=r'$\rho(A_\mathrm{cl})$')
        ax2.axhline(1.0, color='gray', lw=0.5, ls='--')
        ax2.set_ylabel(r'$\rho(A_\mathrm{cl}(p;\bar{K}))$', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)

    if title:
        ax.set_title(title, pad=3)

    if save_path:
        _save(fig, save_path)
    return (fig, ax, ax2) if ax2 is not None else (fig, ax)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, os
    print('=' * 60)
    print('comparison_plots.py — Self-test')
    print('=' * 60)

    rng = np.random.default_rng(42)
    tmpdir = Path(tempfile.mkdtemp())

    # ── Figure 1: synthetic stability data ────────────────────────────────────
    print('\n--- Figure 1: cross-term stability ---')
    N_v, N_eval = 8, 300
    stab_dict = dict(
        rho_open      = 1.0 + rng.uniform(0.01, 0.15, N_v),  # unstable
        rho_diagonal  = 0.85 + rng.uniform(0.0, 0.10, N_v),  # stable (vertex CL)
        rho_joint     = 0.80 + rng.uniform(0.0, 0.15, N_eval),  # joint CL
        grid_pts      = rng.standard_normal((N_eval, 3)),
    )
    p1 = tmpdir / 'fig1_cross_term.pdf'
    fig1, ax1 = plot_cross_term_stability(stab_dict, save_path=p1,
                                           title='Cross-term stability')
    assert p1.exists(), 'Figure 1 not saved'
    print(f'  Figure 1: OK ({p1.stat().st_size // 1024} kB)')

    # ── Figure 2: synthetic convergence ───────────────────────────────────────
    print('\n--- Figure 2: convergence in N ---')
    Ns    = np.array([5, 10, 20, 30, 50, 75, 100, 150, 200])
    J_bar = 120.0; C_true = 20.0; r_true = 0.94
    J_infs = J_bar + C_true * r_true**Ns + rng.normal(0, 0.05, len(Ns))
    J_ric  = 115.0

    from metrics import convergence_rate, detect_convergence
    r_fit, C_fit, Jb_fit = convergence_rate(Ns, J_infs)
    conv_idx = detect_convergence(J_infs, rtol=1e-3, window=3)

    p2 = tmpdir / 'fig2_convergence.pdf'
    fig2, ax2 = plot_convergence(
        Ns, J_infs, J_ric=J_ric,
        rate=r_fit, C=C_fit, J_bar=Jb_fit,
        conv_idx=conv_idx,
        save_path=p2, title='Convergence in $N$',
    )
    assert p2.exists()
    print(f'  Figure 2: OK ({p2.stat().st_size // 1024} kB)')

    # ── Figure 3: synthetic trajectories ─────────────────────────────────────
    print('\n--- Figure 3: trajectory ---')
    from nonlinear_sim import SimResult
    T_sim = 500
    n, m, n_p = 4, 1, 3
    t_ax = np.arange(T_sim + 1) * 0.002

    # Open-loop: diverging alpha
    x_open = np.zeros((T_sim + 1, n))
    x_open[:, 2] = 0.05 * np.exp(0.005 * np.arange(T_sim + 1)) * \
                   np.sin(2 * np.pi * 1.5 * t_ax)
    x_open[:, 0] = 0.01 * np.cumsum(x_open[:, 2]) * 0.002

    # P-RHPG: damping out
    x_prhpg = np.zeros((T_sim + 1, n))
    x_prhpg[:, 2] = 0.05 * np.exp(-0.015 * np.arange(T_sim + 1)) * \
                    np.sin(2 * np.pi * 1.5 * t_ax)
    x_prhpg[:, 0] = 0.01 * np.cumsum(x_prhpg[:, 2]) * 0.002

    def _make_res(x_traj, label):
        return SimResult(
            x_traj=x_traj,
            u_traj=np.zeros((T_sim, m)),
            p_traj=np.zeros((T_sim, n_p)),
            cost=1.0, stable=True, diverged=False,
            div_step=None, settle_step=None,
            elapsed_s=0.1, label=label,
            metadata={'Ts': 0.002},
        )

    res_ol = _make_res(x_open, 'Open-loop')
    res_pg = _make_res(x_prhpg, 'P-RHPG')

    p3 = tmpdir / 'fig3_trajectory.pdf'
    fig3, axes3 = plot_trajectory(
        res_ol, res_pg,
        state_indices=(0, 2),
        state_labels=(r'Plunge $h$ [m]', r'Pitch $\alpha$ [rad]'),
        save_path=p3, title='Closed-loop trajectories',
    )
    assert p3.exists()
    print(f'  Figure 3: OK ({p3.stat().st_size // 1024} kB)')

    # ── Table ─────────────────────────────────────────────────────────────────
    print('\n--- Table: P-RHPG vs LMI ---')
    table_rows = [
        dict(method='P-RHPG', N_v=8,  J_inf=132.4, obj_bound=140.1,
             J_ric=115.0, gap_pct=15.1, solve_time=0.42, feasible=True, stable=True),
        dict(method='P-RHPG', N_v=27, J_inf=121.8, obj_bound=128.3,
             J_ric=115.0, gap_pct=5.9,  solve_time=1.85, feasible=True, stable=True),
        dict(method='LMI-PDC', N_v=8,  J_inf=145.2, obj_bound=172.6,
             J_ric=115.0, gap_pct=26.3, solve_time=0.31, feasible=True, stable=True),
        dict(method='LMI-PDC', N_v=27, J_inf=np.nan, obj_bound=np.nan,
             J_ric=115.0, gap_pct=np.nan, solve_time=4.12, feasible=False, stable=False),
    ]
    p_csv = tmpdir / 'table.csv'
    p_tex = tmpdir / 'table.tex'
    tbl_str, tex_str = format_comparison_table(
        table_rows, save_path_csv=p_csv, save_path_tex=p_tex)
    print(tbl_str)
    assert p_csv.exists() and p_tex.exists()
    print('  Table: OK')

    # ── Cost slice ────────────────────────────────────────────────────────────
    print('\n--- Cost slice 1D ---')
    V_vals = np.linspace(5, 35, 60)
    Vf = 25.56
    pdc_costs = 100 + 2 * (V_vals - 20)**2 + np.where(V_vals > Vf, np.inf, 0)
    ric_costs = 80 + 1.5 * (V_vals - 20)**2 + np.where(V_vals > Vf, np.inf, 0)
    rho_vals  = 0.8 + 0.01 * (V_vals - 20) + np.where(V_vals > Vf, 0.3, 0)

    p4 = tmpdir / 'cost_slice.pdf'
    plot_cost_slice_1d(V_vals, pdc_costs, ric_costs, rho_vals=rho_vals,
                       flutter_speed=Vf, save_path=p4)
    assert p4.exists()
    print(f'  Cost slice: OK ({p4.stat().st_size // 1024} kB)')

    print('\nAll tests passed ✓')
    print(f'\nOutput dir: {tmpdir}')