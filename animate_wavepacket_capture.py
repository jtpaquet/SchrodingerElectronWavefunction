"""
Animate a genuine 3D electron wavepacket -- an isotropic Gaussian of width
`a`, actually displaced to a point in 3D space at distance R from the
nucleus, not smeared into a spherically-symmetric shell -- as R is swept
from far away in to R=0 and back.

This is the real analogue of "an electron approaching a proton": unlike
the l=0 shell family in animate_radial_energy_terms.py, a function of
r=|position| alone can only describe something centered *on* the nucleus;
it cannot describe a wavepacket sitting off to one side. Because the
kinetic energy operator is translation invariant, this doesn't need a
grid at all -- everything is closed form:

    <T>(a)   = 3/(4a^2)                              (exact, independent of R)
    <V>(R,a) = -erf(R/a)/R          (R>0)
             = -2/(a*sqrt(pi))      (R=0, the R->0 limit)

<T> is *exactly* constant here, unlike every l=0 shell variant tried
(plain Gaussian: 3/(4a^2) at R=0 down to 1/(4a^2) far away; the r*exp(...)
node family; a user-proposed (r/a)*exp(-(r-a)^2) family) -- none of which
can be, because none of them are actually a rigid object translating
through ordinary, unbounded 3D space. This one is, so its <T> has to be
exactly what a rigid translation implies: unchanged.

Empirically (see README): with <T> held genuinely fixed like this, E(R)
is monotonically increasing from R=0 for every width `a` tested -- the
minimum is always exactly at the nucleus. There is no artifact-driven
off-center minimum once the comparison is made honestly.

Four panels, axis scales fixed for the whole animation:
  1. A 1D slice of psi along the displacement axis, psi(0,0,z) -- a plain
     Gaussian bump centered at z=R, free to live at any real z (no r>=0
     restriction, because z is an ordinary Cartesian coordinate).
  2. <T> (a flat line -- exactly constant) and total energy E = <T>+<V>
     vs. R, with a marker for the current frame.
  3. A bar chart of <T> vs. |<V>|, log-scale y-axis fixed for the run.
  4. A 3D "electron cloud" -- points sampled from the true isotropic 3D
     Gaussian at its actual displaced position, showing a solid ball that
     simply translates (never hollows into a shell, unlike the l=0 model).

Usage:
    python animate_wavepacket_capture.py --fixed-width 1.0 -o output/wavepacket_capture_a1.0.gif
    python animate_wavepacket_capture.py --fixed-width 0.3 -o output/wavepacket_capture_a0.3.gif
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.special import erf

from schrodinger_terms import reparam_by_energy_variation

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_T = "#2a78d6"
COLOR_V = "#eb6834"
COLOR_E = "#e34948"


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def style_3d_axis(ax):
    ax.set_facecolor(SURFACE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=7)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.zaxis.label.set_color(INK_SECONDARY)


def kinetic_energy(a):
    return 3.0 / (4.0 * a**2)


def potential_energy(R, a):
    """<V>(R,a) = -erf(R/a)/R for R>0, -> -2/(a*sqrt(pi)) as R->0."""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    out = np.where(R < 1e-9, -2.0 / (a * np.sqrt(np.pi)), 0.0)
    mask = R >= 1e-9
    out[mask] = -erf(R[mask] / a) / R[mask]
    return out


def sample_cloud(a, R, n_points, rng):
    """Sample points from the true isotropic 3D Gaussian centered at (0,0,R)."""
    x = rng.normal(0.0, a / np.sqrt(2.0), n_points)
    y = rng.normal(0.0, a / np.sqrt(2.0), n_points)
    z = rng.normal(R, a / np.sqrt(2.0), n_points)
    return x, y, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-width", type=float, default=1.0,
                         help="wavepacket width `a` held fixed while R sweeps (default 1.0, "
                              "the true hydrogen ground-state width)")
    parser.add_argument("--r-max", type=float, default=6.0)
    parser.add_argument("--cloud-points", type=int, default=3000)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    a = args.fixed_width
    output = args.output or f"output/wavepacket_capture_a{a:.2f}.gif"
    rng = np.random.default_rng(args.seed)

    n_fine, n_frames = 400, 60
    R_bg = np.linspace(0.0, args.r_max, n_fine)
    T_bg = np.full_like(R_bg, kinetic_energy(a))
    V_bg = potential_energy(R_bg, a)
    E_bg = T_bg + V_bg

    forward = reparam_by_energy_variation(R_bg, T_bg, V_bg, n_frames)  # starts at R=0
    R_values = np.concatenate([forward, forward[-2:0:-1]])

    z_window = args.r_max + 3 * a
    cloud_window = args.r_max + 3 * a

    fig = plt.figure(figsize=(13, 10.5))
    fig.patch.set_facecolor(SURFACE)
    ax_psi = fig.add_subplot(2, 2, 1)
    ax_energy = fig.add_subplot(2, 2, 2)
    ax_bar = fig.add_subplot(2, 2, 3)
    ax_cloud = fig.add_subplot(2, 2, 4, projection="3d")
    for ax in (ax_psi, ax_energy, ax_bar):
        style_axis(ax)
    style_3d_axis(ax_cloud)
    fig.suptitle(
        f"ψ(x,y,z) = exp(-((x²+y²+(z-R)²)/2a²), a={a:.2f}  --  true 3D displacement, sweeping R",
        color=INK_PRIMARY, fontsize=12, y=0.99,
    )

    # --- Panel 1: 1D slice along the displacement axis ---
    ax_psi.set_xlabel("z (Bohr radii)", color=INK_SECONDARY)
    ax_psi.set_ylabel("ψ(0,0,z) / max", color=INK_SECONDARY)
    ax_psi.set_title("1. Wavepacket slice along z (normalized)", color=INK_PRIMARY, fontsize=11)
    ax_psi.set_xlim(-z_window, z_window)
    ax_psi.set_ylim(0, 1.08)
    (line_psi,) = ax_psi.plot([], [], color=COLOR_T, linewidth=2)
    ax_psi.axvline(0, color=INK_MUTED, linewidth=1, linestyle=":")  # nucleus

    # --- Panel 2: <T> (flat) and total energy vs R ---
    ax_energy.set_xlim(R_bg.min(), R_bg.max())
    y_lo = min(E_bg.min(), T_bg.min())
    y_hi = max(E_bg.max(), T_bg.max())
    e_margin = 0.1 * (y_hi - y_lo + 1e-9)
    ax_energy.set_ylim(y_lo - e_margin, y_hi + e_margin)
    ax_energy.set_xlabel("displacement R", color=INK_SECONDARY)
    ax_energy.set_ylabel("energy (Hartree)", color=INK_SECONDARY)
    ax_energy.set_title("2. Kinetic term (exactly constant) & total energy", color=INK_PRIMARY, fontsize=11)
    ax_energy.axhline(0, color=BASELINE, linewidth=1)
    ax_energy.plot(R_bg, E_bg, color=COLOR_E, linewidth=2, label="E = ⟨T⟩+⟨V⟩")
    ax_energy.plot(R_bg, T_bg, color=COLOR_T, linewidth=2, label="⟨T⟩ = 3/(4a²)")
    ax_energy.legend(loc="center right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    (marker_E,) = ax_energy.plot([], [], "o", color=COLOR_E, markersize=9,
                                  markeredgecolor=SURFACE, markeredgewidth=1.5)
    (marker_T,) = ax_energy.plot([], [], "o", color=COLOR_T, markersize=9,
                                  markeredgecolor=SURFACE, markeredgewidth=1.5)

    # --- Panel 3: bar chart, log-scale, fixed for the run ---
    bar_floor = min(T_bg.min(), np.abs(V_bg[V_bg != 0]).min() if np.any(V_bg != 0) else T_bg.min()) * 0.5
    bar_ceil = max(T_bg.max(), np.abs(V_bg).max()) * 1.5
    ax_bar.set_xlim(-0.6, 1.6)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(["⟨T⟩", "|⟨V⟩|"], color=INK_SECONDARY, fontsize=10)
    ax_bar.set_ylabel("energy (Hartree, log scale)", color=INK_SECONDARY)
    ax_bar.set_yscale("log")
    ax_bar.set_ylim(bar_floor, bar_ceil)
    ax_bar.set_title("3. Kinetic vs. potential", color=INK_PRIMARY, fontsize=11)
    bars = ax_bar.bar([0, 1], [bar_floor, bar_floor], bottom=bar_floor, width=0.6, color=[COLOR_T, COLOR_V])
    ratio_text = ax_bar.text(0.5, 0.94, "", ha="center", va="top", transform=ax_bar.transAxes,
                              color=INK_PRIMARY, fontsize=11)

    # --- Panel 4: 3D electron cloud, a solid ball that only translates ---
    ax_cloud.set_title("4. Electron cloud (true 3D, displaced)", color=INK_PRIMARY, fontsize=11, y=1.0)
    ax_cloud.set_xlabel("x")
    ax_cloud.set_ylabel("y")
    ax_cloud.set_zlabel("z")
    ax_cloud.set_xlim(-cloud_window, cloud_window)
    ax_cloud.set_ylim(-cloud_window, cloud_window)
    ax_cloud.set_zlim(-cloud_window, cloud_window)
    x0, y0, z0 = sample_cloud(a, R_values[0], args.cloud_points, rng)
    cloud = ax_cloud.scatter(x0, y0, z0, color=COLOR_T, alpha=0.45, s=8, linewidths=0)
    ax_cloud.scatter([0], [0], [0], color=COLOR_V, s=40, depthshade=False)  # nucleus marker

    info_text = fig.text(0.5, 0.012, "", ha="center", va="bottom", color=INK_SECONDARY, fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    z = np.linspace(-z_window, z_window, 2000)

    def update(frame_idx):
        R = R_values[frame_idx]
        T = kinetic_energy(a)
        V = potential_energy(np.array([R]), a)[0]
        E = T + V
        ratio = T / abs(V)

        psi_slice = np.exp(-((z - R) ** 2) / (2 * a**2))
        line_psi.set_data(z, psi_slice / psi_slice.max())

        marker_E.set_data([R], [E])
        marker_T.set_data([R], [T])

        bars[0].set_height(T - bar_floor)
        bars[1].set_height(abs(V) - bar_floor)

        xs, ys, zs = sample_cloud(a, R, args.cloud_points, rng)
        cloud._offsets3d = (xs, ys, zs)

        ratio_text.set_text(f"⟨T⟩/|⟨V⟩| = {ratio:.2f}")
        info_text.set_text(f"R = {R:6.3f}    ⟨T⟩ = {T:7.3f}    ⟨V⟩ = {V:7.3f}    E = {E:7.3f}")
        return line_psi, marker_E, marker_T, bars[0], bars[1], ratio_text, cloud, info_text

    anim = FuncAnimation(fig, update, frames=len(R_values), blit=False)
    anim.save(output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {output}  (a={a}, T={kinetic_energy(a):.4f} constant, "
          f"E(0)={E_bg[0]:.4f}, E(R_max)={E_bg[-1]:.4f}, monotonic={np.all(np.diff(E_bg) >= -1e-9)})")


if __name__ == "__main__":
    main()
