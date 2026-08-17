"""
Animate the kinetic vs. Coulomb-potential energy split of a 1D trial
wavefunction under two independent sweeps.

Mode "width" (case 1): the wavefunction stays centered on the nucleus at
the origin, and its width `a` is swept from very large (spread out) down
to very small (tightly localized) and back -- "how spread out is the
electron".

Mode "position" (case 2): the width `a` is held fixed at a somewhat
localized value, and the wavefunction's center `x0` is swept away from
and back across the nucleus -- "how far from the nucleus is the
(equally localized) electron". `x0` is used instead of `r0` (reads as a
Bohr radius) or `a` (already means width here).

Both modes drive the same three panels:
  1. The wavefunction psi(x) itself.
  2. Total energy E = <T> + <V> vs. the swept variable, with a marker
     for the current frame.
  3. A bar chart comparing <T> and |<V>| directly, annotated with the
     ratio <T>/|<V>| (the virial theorem predicts 0.5 at the energy
     minimum, for a pure Coulomb potential).

Usage:
    python animate_energy_terms.py width    --kind hydrogen1s -o output/width_hydrogen1s.gif
    python animate_energy_terms.py width    --kind gaussian    -o output/width_gaussian.gif
    python animate_energy_terms.py position --kind gaussian    -o output/position_gaussian.gif
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from schrodinger_terms import (
    energy_terms,
    make_grid,
    potential,
    sweep_positions,
    sweep_widths,
    wavefunction,
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
    "hydrogen1s": "exp(-|x-x0|/a)",
    "gaussian": "exp(-(x-x0)^2/2a^2)",
}


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def run_width_mode(args, x, V):
    a_max, a_min, n = 4.0, 0.35, 60
    forward = np.linspace(a_max, a_min, n)  # start from a very large width
    a_values = np.concatenate([forward, forward[-2:0:-1]])  # ping-pong loop
    T_bg, V_bg, E_bg = sweep_widths(forward, x, V, args.kind)
    sweep_bg = forward
    a_star = sweep_bg[np.argmin(E_bg)]

    def frame_state(val):
        psi = wavefunction(x, val, args.kind, x0=0.0)
        T, Vexp, _, _ = energy_terms(x, psi, V)
        return psi, T, Vexp

    title = f"ψ(x;a) = {KIND_FORMULA[args.kind]}, centered on the nucleus  --  sweeping width a"
    xlabel = "width parameter a"
    # In true 3D hydrogen (no regularization needed -- see README) the
    # virial theorem 2<T>=|<V>| holds exactly at this minimum. Here the
    # 1D Coulomb term needed soft-core regularization to stay finite,
    # which breaks the exact scaling the theorem relies on, so the
    # on-chart ratio at this point is only close to 0.5, not exactly it.
    extremum_label = "energy minimum"
    return a_values, sweep_bg, T_bg, V_bg, E_bg, a_star, frame_state, title, xlabel, extremum_label


def run_position_mode(args, x, V):
    a_fixed = args.fixed_width
    x0_max, n = 8.0, 60
    forward = np.linspace(-x0_max, x0_max, n)  # start far from the nucleus
    x0_values = np.concatenate([forward, forward[-2:0:-1]])
    T_bg, V_bg, E_bg = sweep_positions(forward, x, V, a_fixed, args.kind)
    sweep_bg = forward
    x0_star = sweep_bg[np.argmin(E_bg)]

    def frame_state(val):
        psi = wavefunction(x, a_fixed, args.kind, x0=val)
        T, Vexp, _, _ = energy_terms(x, psi, V)
        return psi, T, Vexp

    title = (
        f"ψ(x;x0) = {KIND_FORMULA[args.kind]}, fixed width a={a_fixed:.2f} (localized electron)"
        "  --  sweeping center x0"
    )
    xlabel = "center x0 (Bohr radii from the nucleus)"
    # Note: the virial theorem (2<T>=|<V>|) governs *scaling* (the width
    # sweep above), not translation -- so it does not apply at this
    # energy minimum. Translating a fixed-shape wavefunction leaves <T>
    # essentially unchanged; only <V> varies with x0.
    extremum_label = "closest approach (min E)"
    return x0_values, sweep_bg, T_bg, V_bg, E_bg, x0_star, frame_state, title, xlabel, extremum_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["width", "position"])
    parser.add_argument("--kind", choices=["hydrogen1s", "gaussian"], default="gaussian")
    parser.add_argument("--fixed-width", type=float, default=1.0,
                         help="width `a` held fixed in position mode (default 1.0, the "
                              "true 3D hydrogen ground-state Bohr-radius-optimal width)")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    output = args.output or f"output/{args.mode}_{args.kind}.gif"

    x = make_grid(L=40.0, N=4000)
    V = potential(x)

    if args.mode == "width":
        (sweep_values, sweep_bg, T_bg, V_bg, E_bg, star, frame_state, title, xlabel,
         extremum_label) = run_width_mode(args, x, V)
    else:
        (sweep_values, sweep_bg, T_bg, V_bg, E_bg, star, frame_state, title, xlabel,
         extremum_label) = run_position_mode(args, x, V)

    # Fixed y-limits for panel 1, sized to the tallest wavefunction in the sweep.
    tallest = max(frame_state(v)[0].max() for v in (sweep_bg[0], sweep_bg[-1], sweep_bg[len(sweep_bg) // 2]))
    psi_ylim = tallest * 1.15

    fig, (ax_psi, ax_energy, ax_bar) = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_psi, ax_energy, ax_bar):
        style_axis(ax)
    fig.suptitle(title, color=INK_PRIMARY, fontsize=12, y=0.99)

    # --- Panel 1: the wavefunction ---
    ax_psi.set_xlim(-12, 12)
    ax_psi.set_ylim(0, psi_ylim)
    ax_psi.set_xlabel("x (Bohr radii)", color=INK_SECONDARY)
    ax_psi.set_ylabel("ψ(x)", color=INK_SECONDARY)
    ax_psi.set_title("1. Wavefunction", color=INK_PRIMARY, fontsize=11)
    (line_psi,) = ax_psi.plot([], [], color=COLOR_T, linewidth=2)
    ax_psi.axvline(0, color=INK_MUTED, linewidth=1, linestyle=":")  # nucleus

    # --- Panel 2: total energy vs. swept variable ---
    ax_energy.set_xlim(sweep_bg.min(), sweep_bg.max())
    e_margin = 0.1 * (E_bg.max() - E_bg.min() + 1e-9)
    ax_energy.set_ylim(E_bg.min() - e_margin, E_bg.max() + e_margin)
    ax_energy.set_xlabel(xlabel, color=INK_SECONDARY)
    ax_energy.set_ylabel("E (Hartree)", color=INK_SECONDARY)
    ax_energy.set_title("2. Total energy E = ⟨T⟩+⟨V⟩", color=INK_PRIMARY, fontsize=11)
    ax_energy.axhline(0, color=BASELINE, linewidth=1)
    ax_energy.axvline(star, color=INK_MUTED, linewidth=1, linestyle=":")
    ax_energy.plot(sweep_bg, E_bg, color=COLOR_E, linewidth=2)
    (marker_E,) = ax_energy.plot([], [], "o", color=COLOR_E, markersize=9,
                                  markeredgecolor=SURFACE, markeredgewidth=1.5)

    # --- Panel 3: bar chart comparing <T> and |<V>| ---
    bar_ylim = max(T_bg.max(), np.abs(V_bg).max()) * 1.2
    ax_bar.set_xlim(-0.6, 1.6)
    ax_bar.set_ylim(0, bar_ylim)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(["⟨T⟩", "|⟨V⟩|"], color=INK_SECONDARY, fontsize=10)
    ax_bar.set_ylabel("energy (Hartree)", color=INK_SECONDARY)
    ax_bar.set_title("3. Kinetic vs. potential", color=INK_PRIMARY, fontsize=11)
    bars = ax_bar.bar([0, 1], [0, 0], width=0.6, color=[COLOR_T, COLOR_V])
    ratio_text = ax_bar.text(0.5, bar_ylim * 0.94, "", ha="center", va="top",
                              color=INK_PRIMARY, fontsize=11)

    info_text = fig.text(0.5, 0.015, "", ha="center", va="bottom", color=INK_SECONDARY, fontsize=10)

    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    def update(frame_idx):
        val = sweep_values[frame_idx]
        psi, T, Vexp = frame_state(val)
        E = T + Vexp
        ratio = T / abs(Vexp)

        line_psi.set_data(x, psi)
        marker_E.set_data([val], [E])
        bars[0].set_height(T)
        bars[1].set_height(abs(Vexp))

        note = f" ({extremum_label})" if abs(val - star) < abs(sweep_bg[1] - sweep_bg[0]) else ""
        ratio_text.set_text(f"⟨T⟩/|⟨V⟩| = {ratio:.2f}{note}")
        info_text.set_text(
            f"{xlabel.split(' ')[0]} = {val:5.2f}    ⟨T⟩ = {T:6.3f}    ⟨V⟩ = {Vexp:7.3f}    E = {E:7.3f}"
        )
        return line_psi, marker_E, bars[0], bars[1], ratio_text, info_text

    anim = FuncAnimation(fig, update, frames=len(sweep_values), blit=False)
    anim.save(output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {output}  (extremum at {star:.3f}, min/max E = {E_bg.min():.4f}/{E_bg.max():.4f})")


if __name__ == "__main__":
    main()
