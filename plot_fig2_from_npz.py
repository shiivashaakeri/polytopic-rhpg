"""
Regenerate Figure 2 from saved NPZ files — no recomputation needed.

Two-panel layout:
  Left  — full log scale, N=5 to 200
  Right — linear zoom on convergence region, N=50 to 1000

Usage:
    python plot_fig2_from_npz.py
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT    = Path(__file__).parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
d_zero = np.load(ROOT / 'results' / 'exp2_zero.npz')
d_are  = np.load(ROOT / 'results' / 'exp2_are.npz')

Ns_z   = d_zero['Ns'];    JNs_z = d_zero['J_Ns']
Ns_a   = d_are['Ns'];     JNs_a = d_are['J_Ns']
J_bar  = float(d_zero['J_bar'])
J_feas = float(d_zero['J_feas'])

print(f'  J_bar  = {J_bar:.6e}')
print(f'  J_feas = {J_feas:.6e}')

# Left panel: N <= 200
mask_z_l = Ns_z <= 200;  Ns_z_l = Ns_z[mask_z_l];  JNs_z_l = JNs_z[mask_z_l]
mask_a_l = Ns_a <= 200;  Ns_a_l = Ns_a[mask_a_l];  JNs_a_l = JNs_a[mask_a_l]

# Right panel: all N (including 500, 1000), starting from N=50
mask_z_r = Ns_z >= 50;  Ns_z_r = Ns_z[mask_z_r];  JNs_z_r = JNs_z[mask_z_r]
mask_a_r = Ns_a >= 50;  Ns_a_r = Ns_a[mask_a_r];  JNs_a_r = JNs_a[mask_a_r]

# ── Style ─────────────────────────────────────────────────────────────────────
IEEE_FULL = 7.16
IEEE_H    = 2.4
DPI       = 300

C_ZERO = '#1f77b4'
C_ARE  = '#d62728'
C_JBAR = 'k'
C_FEAS = '#9467bd'

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.size'       : 8,
    'axes.labelsize'  : 8,
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

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(IEEE_FULL, IEEE_H),
                                  gridspec_kw={'width_ratios': [1, 1.2]})

# shared reference line helper
def ref_lines(ax, lw=1.0):
    ax.axhline(J_bar,  color=C_JBAR, lw=lw, ls=':', zorder=3)
    ax.axhline(J_feas, color=C_FEAS, lw=lw, ls='-', zorder=3)

# ── Left panel: full log scale, N=5..200 ─────────────────────────────────────
ax_l.semilogy(Ns_z_l, JNs_z_l, color=C_ZERO, marker='s', ms=4, lw=1.4, zorder=4,
              label=r'$J_N^*$, $Q_N{=}0$ (non-decr.)')
ax_l.semilogy(Ns_a_l, JNs_a_l, color=C_ARE,  marker='o', ms=4, lw=1.4, zorder=4,
              label=r'$J_N^*$, $Q_N{=}P_{\mathrm{are}}$ (non-incr.)')
ref_lines(ax_l)

ax_l.set_xlabel(r'Horizon $N$')
ax_l.set_ylabel(r'$J_N^*$')
ax_l.set_xlim(-5, 210)
ax_l.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax_l.set_title('(a) Full range', fontsize=8, pad=3)

# Legend top-left — red curve is coming down from top-left, blue is bottom-left
# so top-right is cleanest
ax_l.legend(
    loc='lower right', framealpha=0.9, ncol=1,
    handles=[
        plt.Line2D([0],[0], color=C_ZERO, marker='s', ms=4, lw=1.4),
        plt.Line2D([0],[0], color=C_ARE,  marker='o', ms=4, lw=1.4),
        plt.Line2D([0],[0], color=C_JBAR, lw=1.0, ls=':'),
        plt.Line2D([0],[0], color=C_FEAS, lw=1.0, ls='-'),
    ],
    labels=[
        r'$J_N^*$, $Q_N{=}0$ (non-decr.)',
        r'$J_N^*$, $Q_N{=}P_{\mathrm{are}}$ (non-incr.)',
        r'$\bar{J}$ (limit, $N{=}1000$)',
        r'$J_\infty(\mathbf{K}^{\mathrm{feas}})$ (Thm. 2-ii)',
    ],
)

# ── Right panel: linear, N=50..1000 ──────────────────────────────────────────
ax_r.plot(Ns_z_r, JNs_z_r, color=C_ZERO, marker='s', ms=4, lw=1.4, zorder=4)
ax_r.plot(Ns_a_r, JNs_a_r, color=C_ARE,  marker='o', ms=4, lw=1.4, zorder=4)
ref_lines(ax_r)

# Shade squeeze region between the two curves
min_len = min(len(Ns_z_r), len(Ns_a_r))
ax_r.fill_between(Ns_z_r[:min_len], JNs_z_r[:min_len], JNs_a_r[:min_len],
                  alpha=0.10, color='gray', zorder=1)

# y limits: just below min of blue, just above J_feas
y_lo_r = min(JNs_z_r.min(), JNs_a_r.min()) * 0.998
y_hi_r = J_feas * 1.015
ax_r.set_ylim(y_lo_r, y_hi_r)
ax_r.set_xlim(40, 1050)

# x-axis: log scale to compress the 500,1000 points
ax_r.set_xscale('log')
ax_r.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax_r.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
ax_r.xaxis.set_minor_locator(mticker.NullLocator())

ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{x/1e5:.2f}'
))
ax_r.set_xlabel(r'Horizon $N$ (log scale)')
ax_r.set_ylabel(r'$J_N^* \;[\times 10^5]$')
ax_r.set_title('(b) Convergence region', fontsize=8, pad=3)



fig.tight_layout(pad=1.2)
out = FIG_DIR / 'fig2_convergence.pdf'
fig.savefig(out, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f'Saved -> {out}')

# ── Also save as two separate single-column figures ───────────────────────────
IEEE_HALF = 3.5
IEEE_H    = 2.4

fig_a, ax_a = plt.subplots(figsize=(IEEE_HALF, IEEE_H))
ax_a.semilogy(Ns_z_l, JNs_z_l, color=C_ZERO, marker='s', ms=4, lw=1.4,
              label=r'$J_N^*$, $Q_N{=}0$ (non-decr.)', zorder=4)
ax_a.semilogy(Ns_a_l, JNs_a_l, color=C_ARE,  marker='o', ms=4, lw=1.4,
              label=r'$J_N^*$, $Q_N{=}P_{\mathrm{are}}$ (non-incr.)', zorder=4)
ax_a.axhline(J_bar,  color=C_JBAR, lw=1.0, ls=':', zorder=3,
             label=r'$\bar{J}$ (limit, $N{=}1000$)')
ax_a.axhline(J_feas, color=C_FEAS, lw=1.0, ls='-', zorder=3,
             label=r'$J_\infty(\mathbf{K}^{\mathrm{feas}})$')
ax_a.set_xlabel(r'Horizon $N$')
ax_a.set_ylabel(r'$J_N^*$')
ax_a.set_xlim(-5, 210)
ax_a.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax_a.legend(loc='lower right', framealpha=0.9, ncol=1)
fig_a.tight_layout()
fig_a.savefig(FIG_DIR / 'fig2a_full.pdf', dpi=DPI, bbox_inches='tight')
plt.close(fig_a)
print(f'Saved -> {FIG_DIR}/fig2a_full.pdf')

# Right panel only
fig_b, ax_b = plt.subplots(figsize=(IEEE_HALF, IEEE_H))
ax_b.plot(Ns_z_r, JNs_z_r, color=C_ZERO, marker='s', ms=4, lw=1.4, zorder=4)
ax_b.plot(Ns_a_r, JNs_a_r, color=C_ARE,  marker='o', ms=4, lw=1.4, zorder=4)
ax_b.axhline(J_bar,  color=C_JBAR, lw=1.0, ls=':', zorder=3)
ax_b.axhline(J_feas, color=C_FEAS, lw=1.0, ls='-', zorder=3)
min_len = min(len(Ns_z_r), len(Ns_a_r))
ax_b.fill_between(Ns_z_r[:min_len], JNs_z_r[:min_len], JNs_a_r[:min_len],
                  alpha=0.10, color='gray', zorder=1)
y_lo_b = min(JNs_z_r.min(), JNs_a_r.min()) * 0.998
ax_b.set_ylim(y_lo_b, J_feas * 1.015)
ax_b.set_xlim(40, 1050)
ax_b.set_xscale('log')
ax_b.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax_b.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
ax_b.xaxis.set_minor_locator(mticker.NullLocator())
ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e5:.2f}'))
ax_b.set_xlabel(r'Horizon $N$ (log scale)')
ax_b.set_ylabel(r'$J_N^* \;[\times 10^5]$')
ax_b.legend(
    handles=[
        plt.Line2D([0],[0], color=C_ZERO, marker='s', ms=4, lw=1.4),
        plt.Line2D([0],[0], color=C_ARE,  marker='o', ms=4, lw=1.4),
        plt.Line2D([0],[0], color=C_JBAR, lw=1.0, ls=':'),
        plt.Line2D([0],[0], color=C_FEAS, lw=1.0, ls='-'),
    ],
    labels=[
        r'$Q_N{=}0$ (non-decr.)',
        r'$Q_N{=}P_{\mathrm{are}}$ (non-incr.)',
        r'$\bar{J}$ (limit)',
        r'$J_\infty(\mathbf{K}^{\mathrm{feas}})$',
    ],
    loc='lower right', framealpha=0.9, fontsize=6.5,
)
fig_b.tight_layout()
fig_b.savefig(FIG_DIR / 'fig2b_zoom.pdf', dpi=DPI, bbox_inches='tight')
plt.close(fig_b)
print(f'Saved -> {FIG_DIR}/fig2b_zoom.pdf')
