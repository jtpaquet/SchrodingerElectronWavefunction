"""
Animate the kinetic vs. Coulomb-potential energy split of a genuine 3D
hydrogen-atom trial wavefunction, restricted to l=0 (spherically
symmetric, no theta/phi dependence). Because the integrals use the
actual spherical volume element 4*pi*r^2*dr, -1/r is integrable at the
origin and no soft-core regularization is needed (contrast with
animate_energy_terms.py's 1D-line model, which does need it -- see
README).

Mode "width": the radial bump stays centered on the nucleus (r0=0), and
its width `a` is swept from very large down to very small and back, well
past the energy-minimizing width, to show the ~1/a^2 kinetic-energy cost
of confinement.

Mode "shell": the width `a` is held fixed at a somewhat localized value,
and the bump's center r0 -- the radius of a spherical shell of fixed
thickness -- is swept outward from the nucleus and back. There's no way
to "translate" a spherically symmetric function off-center without
breaking the symmetry, so this (not a Cartesian shift) is the 3D
equivalent of the flat-line model's position sweep. Unlike that flat
model, this one can show a genuine energy minimum away from the
nucleus: for a rigid shell thin enough to be clipped by the r=0
boundary when centered there, moving it outward trades a shrinking
potential-energy benefit against a shrinking "boundary truncation"
kinetic-energy penalty -- see README for the full explanation and why
it's a *different* mechanism from the textbook 4*pi*r^2 probability
peak.

Four panels, all with axis scales fixed for the whole animation (no
per-frame rescaling) so only the curves/points move, not the frame:
  1. The radial wavefunction, normalized to psi/max(psi) each frame --
     its raw amplitude isn't the interesting part and varies by orders
     of magnitude, so normalizing keeps the y-axis fixed at [0,1] and
     lets the x-extent alone show the change in spread.
  2. <T> and total energy E = <T> + <V> vs. the swept variable, log-x in
     width mode so the 1/a^2 blow-up and the flat tail are both visible.
  3. A bar chart comparing <T> and |<V>|, log-scale y-axis (fixed for
     the whole run) since the two span orders of magnitude in width
     mode -- a fixed linear scale would make most frames unreadable.
  4. A 3D "electron cloud" -- points sampled from |psi(r)|^2 in 3D
     (uniform on the sphere at each sampled radius, exact for l=0),
     drawn with low alpha so density reads visually.

Usage:
    python animate_radial_energy_terms.py width --kind hydrogen1s -o output/radial_width_hydrogen1s.gif
    python animate_radial_energy_terms.py width --kind gaussian    -o output/radial_width_gaussian.gif
    python animate_radial_energy_terms.py shell  --kind gaussian    -o output/radial_shell_gaussian.gif
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from schrodinger_terms import (
    make_radial_grid,
    radial_energy_terms,
    radial_wavefunction,
    reparam_by_energy_variation,
    sample_electron_cloud,
    sweep_radial_shells,
    sweep_radial_widths,
)

# Reference palette (validated categorical order) -- color follows the
# entity (T, V, E) consistently across every panel.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_T = "#2a78d6"   # slot 1, blue
COLOR_V = "#eb6834"   # slot 2, orange
COLOR_E = "#e34948"   # slot 8, red

KIND_FORMULA = {
    "hydrogen1s": "exp(-|r-r0|/a)",
    "gaussian": "exp(-(r-r0)^2/2a^2)",
}


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def style_3d_axis(ax):
    ax.set_facecolor(SURFACE)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
        pane.line.set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=7)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.zaxis.label.set_color(INK_SECONDARY)


def run_width_mode(args, r):
    a_max, a_min, n_frames, n_fine = 4.0, 0.06, 70, 500
    sweep_bg = np.geomspace(a_min, a_max, n_fine)
    T_bg, V_bg, E_bg = sweep_radial_widths(sweep_bg, r, args.kind)
    a_star = sweep_bg[np.argmin(E_bg)]

    frames_ascending = reparam_by_energy_variation(sweep_bg, T_bg, V_bg, n_frames)
    forward = frames_ascending[::-1]  # start from a very large width
    a_values = np.concatenate([forward, forward[-2:0:-1]])

    def frame_state(val):
        psi = radial_wavefunction(r, val, args.kind, r0=0.0)
        T, Vexp, _, _ = radial_energy_terms(r, psi)
        return psi, T, Vexp

    title = f"ψ(r;a) = {KIND_FORMULA[args.kind]}, l=0, centered on the nucleus  --  sweeping width a"
    xlabel = "width parameter a"
    extremum_label = "energy minimum (virial: 2⟨T⟩=|⟨V⟩| exactly, no softening needed)"
    return dict(
        sweep_values=a_values, sweep_bg=sweep_bg, T_bg=T_bg, V_bg=V_bg, E_bg=E_bg, star=a_star,
        frame_state=frame_state, title=title, xlabel=xlabel, extremum_label=extremum_label,
        log_x=True, psi_window=15.0, cloud_window=15.0,
    )


def run_shell_mode(args, r):
    a_fixed = args.fixed_width
    r0_max, n_frames, n_fine = 3.5, 60, 400
    sweep_bg = np.linspace(0.0, r0_max, n_fine)
    T_bg, V_bg, E_bg = sweep_radial_shells(sweep_bg, r, a_fixed, args.kind)
    r0_star = sweep_bg[np.argmin(E_bg)]

    forward = reparam_by_energy_variation(sweep_bg, T_bg, V_bg, n_frames)  # starts at r0=0
    r0_values = np.concatenate([forward, forward[-2:0:-1]])

    def frame_state(val):
        psi = radial_wavefunction(r, a_fixed, args.kind, r0=val)
        T, Vexp, _, _ = radial_energy_terms(r, psi)
        return psi, T, Vexp

    title = (
        f"ψ(r;r0) = {KIND_FORMULA[args.kind]}, l=0, fixed thickness a={a_fixed:.2f}"
        "  --  sweeping shell radius r0"
    )
    xlabel = "shell radius r0"
    extremum_label = "energy minimum (off the nucleus)"
    return dict(
        sweep_values=r0_values, sweep_bg=sweep_bg, T_bg=T_bg, V_bg=V_bg, E_bg=E_bg, star=r0_star,
        frame_state=frame_state, title=title, xlabel=xlabel, extremum_label=extremum_label,
        log_x=False, psi_window=r0_max + 3 * a_fixed, cloud_window=r0_max + 4 * a_fixed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["width", "shell"])
    parser.add_argument("--kind", choices=["hydrogen1s", "gaussian"], default="gaussian")
    parser.add_argument("--fixed-width", type=float, default=0.2,
                         help="thickness `a` held fixed in shell mode (default 0.2: thin enough "
                              "that the r=0 boundary-clipping penalty dominates and the shell's "
                              "energy minimum sits clearly off the nucleus -- see README; this "
                              "narrow a shell is unbound, E>0 throughout, unlike --fixed-width 0.7)")
    parser.add_argument("--cloud-points", type=int, default=3000)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output or f"output/radial_{args.mode}_{args.kind}.gif"

    r = make_radial_grid(r_max=25.0, N=20000) if args.mode == "width" else make_radial_grid(r_max=20.0, N=8000)
    rng = np.random.default_rng(args.seed)

    run = (run_width_mode if args.mode == "width" else run_shell_mode)(args, r)
    sweep_values, sweep_bg = run["sweep_values"], run["sweep_bg"]
    T_bg, V_bg, E_bg, star = run["T_bg"], run["V_bg"], run["E_bg"], run["star"]
    frame_state, title, xlabel, extremum_label = (
        run["frame_state"], run["title"], run["xlabel"], run["extremum_label"]
    )

    fig = plt.figure(figsize=(13, 10.5))
    fig.patch.set_facecolor(SURFACE)
    ax_psi = fig.add_subplot(2, 2, 1)
    ax_energy = fig.add_subplot(2, 2, 2)
    ax_bar = fig.add_subplot(2, 2, 3)
    ax_cloud = fig.add_subplot(2, 2, 4, projection="3d")
    for ax in (ax_psi, ax_energy, ax_bar):
        style_axis(ax)
    style_3d_axis(ax_cloud)
    fig.suptitle(title, color=INK_PRIMARY, fontsize=12, y=0.99)

    # --- Panel 1: the radial wavefunction, normalized to psi/max(psi) so
    # the y-axis stays fixed at [0,1] regardless of the raw amplitude ---
    ax_psi.set_xlabel("r (Bohr radii)", color=INK_SECONDARY)
    ax_psi.set_ylabel("ψ(r) / max(ψ)", color=INK_SECONDARY)
    ax_psi.set_title("1. Radial wavefunction (normalized)", color=INK_PRIMARY, fontsize=11)
    ax_psi.set_xlim(0, run["psi_window"])
    ax_psi.set_ylim(0, 1.08)
    (line_psi,) = ax_psi.plot([], [], color=COLOR_T, linewidth=2)
    ax_psi.axvline(0, color=INK_MUTED, linewidth=1, linestyle=":")  # nucleus

    # --- Panel 2: <T> and total energy vs. swept variable ---
    ax_energy.set_xlim(max(sweep_bg.min(), 1e-3) if run["log_x"] else sweep_bg.min(), sweep_bg.max())
    if run["log_x"]:
        ax_energy.set_xscale("log")
    y_lo = min(E_bg.min(), T_bg.min())
    y_hi = max(E_bg.max(), T_bg.max())
    e_margin = 0.1 * (y_hi - y_lo + 1e-9)
    ax_energy.set_ylim(y_lo - e_margin, y_hi + e_margin)
    ax_energy.set_xlabel(xlabel, color=INK_SECONDARY)
    ax_energy.set_ylabel("energy (Hartree)", color=INK_SECONDARY)
    ax_energy.set_title("2. Kinetic term & total energy", color=INK_PRIMARY, fontsize=11)
    ax_energy.axhline(0, color=BASELINE, linewidth=1)
    ax_energy.axvline(star, color=INK_MUTED, linewidth=1, linestyle=":")
    ax_energy.plot(sweep_bg, E_bg, color=COLOR_E, linewidth=2, label="E = ⟨T⟩+⟨V⟩")
    ax_energy.plot(sweep_bg, T_bg, color=COLOR_T, linewidth=2, label="⟨T⟩")
    ax_energy.legend(loc="upper center", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    (marker_E,) = ax_energy.plot([], [], "o", color=COLOR_E, markersize=9,
                                  markeredgecolor=SURFACE, markeredgewidth=1.5)
    (marker_T,) = ax_energy.plot([], [], "o", color=COLOR_T, markersize=9,
                                  markeredgecolor=SURFACE, markeredgewidth=1.5)

    # --- Panel 3: bar chart comparing <T> and |<V>|, log-scale y-axis
    # fixed for the whole run -- <T> and |<V>| span orders of magnitude
    # in width mode, so a fixed linear scale would make most frames
    # unreadable, and a per-frame-rescaled one would defeat "fixed axes".
    bar_floor = min(T_bg.min(), np.abs(V_bg).min()) * 0.5
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

    # --- Panel 4: 3D electron cloud, cube fixed for the whole run ---
    ax_cloud.set_title("4. Electron cloud (|ψ|² in 3D)", color=INK_PRIMARY, fontsize=11, y=1.0)
    ax_cloud.set_xlabel("x")
    ax_cloud.set_ylabel("y")
    ax_cloud.set_zlabel("z")
    cloud_half = run["cloud_window"]
    ax_cloud.set_xlim(-cloud_half, cloud_half)
    ax_cloud.set_ylim(-cloud_half, cloud_half)
    ax_cloud.set_zlim(-cloud_half, cloud_half)
    x0, y0, z0 = sample_electron_cloud(r, frame_state(sweep_bg[0])[0], args.cloud_points, rng)
    cloud = ax_cloud.scatter(x0, y0, z0, color=COLOR_T, alpha=0.45, s=8, linewidths=0)
    ax_cloud.scatter([0], [0], [0], color=COLOR_V, s=40, depthshade=False)  # nucleus marker

    info_text = fig.text(0.5, 0.012, "", ha="center", va="bottom", color=INK_SECONDARY, fontsize=10)

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    near_star_tol = np.median(np.abs(np.diff(sweep_values))) * 0.75

    def update(frame_idx):
        val = sweep_values[frame_idx]
        psi, T, Vexp = frame_state(val)
        E = T + Vexp
        ratio = T / abs(Vexp)

        line_psi.set_data(r, psi / psi.max())

        marker_E.set_data([val], [E])
        marker_T.set_data([val], [T])

        bars[0].set_height(T - bar_floor)
        bars[1].set_height(abs(Vexp) - bar_floor)

        xs, ys, zs = sample_electron_cloud(r, psi, args.cloud_points, rng)
        cloud._offsets3d = (xs, ys, zs)

        note = f" ({extremum_label})" if abs(val - star) < near_star_tol else ""
        ratio_text.set_text(f"⟨T⟩/|⟨V⟩| = {ratio:.2f}{note}")
        info_text.set_text(
            f"{xlabel.split(' ')[0]} = {val:6.3f}    ⟨T⟩ = {T:7.3f}    ⟨V⟩ = {Vexp:7.3f}    E = {E:7.3f}"
        )
        return line_psi, marker_E, marker_T, bars[0], bars[1], ratio_text, cloud, info_text

    anim = FuncAnimation(fig, update, frames=len(sweep_values), blit=False)
    anim.save(output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {output}  (extremum at {star:.4f}, min/max E = {E_bg.min():.4f}/{E_bg.max():.4f}, "
          f"max <T> = {T_bg.max():.3f})")


if __name__ == "__main__":
    main()
