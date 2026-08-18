"""
Genuine real-time evolution of an l=0 (spherically symmetric) hydrogen-atom
wavefunction under the time-dependent Schrodinger equation, starting from an
off-center or too-wide/too-narrow bump like the ones swept in
animate_radial_energy_terms.py -- and showing what actually happens to it.

Physics point this animation exists to make: unitary time evolution does
*not* relax a wavepacket into the ground state. Expand the initial state in
the exact hydrogen ns eigenbasis, psi(r,0) = sum_n c_n psi_n(r). Each term
just picks up a phase under time evolution, c_n -> c_n * exp(-i*E_n*t), so
|c_n|^2 -- the population in eigenstate n -- is exactly constant for all
time. The wavepacket breathes, spreads, and partially revives (energy sloshes
between <T> and <V>), but the total energy E=<T>+<V> and every |c_n|^2 stay
flat. Only *imaginary*-time propagation (e^{-H*tau} instead of e^{-iHt}, not
implemented here -- not physical evolution, just a ground-state-finding
numerical trick) would make it converge to n=1.

Numerics: the standard l=0 radial substitution u(r,t) = sqrt(4*pi)*r*psi(r,t)
turns the full 3D equation i*d(psi)/dt = -1/2*Laplacian(psi) + V(r)*psi into
a plain 1D equation i*du/dt = -1/2*u'' + V(r)*u on r in [0, r_max], with
Dirichlet boundaries u(0)=0 (regularity at the origin -- forced by the
substitution itself, not imposed) and u(r_max)=0 (hard wall; r_max is chosen
generous enough that reflections off it don't contaminate the run). This
substitution is exact for l=0 (the r^2/r^-2 pieces of the 3D Laplacian cancel
identically -- see README/schrodinger_terms.py for the l=0 reduction used
throughout this repo), and it makes |u(r,t)|^2 *equal* to the radial
probability density P(r,t) = 4*pi*r^2*|psi(r,t)|^2 with no extra factors, and
<T> = 1/2 * Integral |u'(r)|^2 dr, <V> = Integral V(r)|u(r)|^2 dr -- both
already including the spherical measure, no separate 4*pi*r^2 factor needed
(this is the same identity that makes radial_energy_terms's 2*pi*r^2*psi'^2
kinetic density correct, applied to complex u instead of real psi -- see
schrodinger_terms.py).

Crank-Nicolson time-stepping, (I + i*dt/2*H) u^{n+1} = (I - i*dt/2*H) u^n, is
used because it's *exactly* unitary (a Cayley transform of a Hermitian H) at
any dt -- norm and, up to spatial discretization error, every |c_n|^2 are
conserved by construction, not just approximately by using a small enough
step. That's the cleanest way to demonstrate energy/population conservation
numerically: any drift you see is discretization error, not a real physical
decay channel. The Hamiltonian and its Cayley factors don't depend on time,
so the factorization is done once (scipy's sparse LU) and reused for every
step.

Four panels, all axis-scale-fixed for the whole run:
  1. P(r,t) = |u(r,t)|^2 -- the radial probability density, breathing/
     sloshing in place.
  2. <T>(t), <V>(t) (both oscillate, trading energy back and forth) and
     E(t)=<T>+<V> (flat -- energy conservation).
  3. |c_n(t)|^2 for n=1..--max-n -- population in each exact hydrogen ns
     eigenstate. Flat lines: the headline result of this script.
  4. A 3D "electron cloud" sampled from P(r,t), showing the same breathing
     seen in panel 1 as an actual 3D blob.

Usage:
    python animate_radial_time_evolution.py --width 2.0 -o output/radial_time_evolution_wide.gif
    python animate_radial_time_evolution.py --width 0.6 --r0 4.0 -o output/radial_time_evolution_offcenter.gif
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.animation import FuncAnimation, PillowWriter

from schrodinger_terms import radial_wavefunction

# numpy >=2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_T = "#2a78d6"
COLOR_V = "#eb6834"
COLOR_P = "#1baf7a"
COLOR_E = "#e34948"
POP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a5fd6", "#c9a227"]


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


def build_cayley_factors(r_interior, dr, dt):
    """Tridiagonal H = -1/2 d^2/dr^2 + V(r) on the interior grid (Dirichlet
    zero at both ends), and the two Cayley-transform matrices for
    Crank-Nicolson: A u^{n+1} = B u^n, A = I + i*dt/2*H, B = I - i*dt/2*H."""
    V = -1.0 / r_interior
    main = 1.0 / dr**2 + V
    off = -0.5 / dr**2 * np.ones(len(r_interior) - 1)
    H = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csc")
    M = len(r_interior)
    I = sp.identity(M, format="csc", dtype=complex)
    A = (I + 1j * dt / 2 * H).tocsc()
    B = (I - 1j * dt / 2 * H).tocsc()
    return A, B, V


def kinetic_potential(u_full, r, V_full):
    du_dr = np.gradient(u_full, r)
    T = 0.5 * _trapz(np.abs(du_dr) ** 2, r)
    Vexp = np.real(_trapz(V_full * np.abs(u_full) ** 2, r))
    return T, Vexp


def sample_cloud_from_density(r, density, n_points, rng):
    """Sample 3D points whose radii follow `density` (any positive multiple
    of the true radial density works -- the CDF normalizes it away) and
    whose direction is uniform on the sphere (exact for l=0)."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(r))])
    cdf /= cdf[-1]
    u = rng.random(n_points)
    radii = np.interp(u, cdf, r)
    cos_theta = rng.uniform(-1.0, 1.0, n_points)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.uniform(0.0, 2 * np.pi, n_points)
    x = radii * sin_theta * np.cos(phi)
    y = radii * sin_theta * np.sin(phi)
    z = radii * cos_theta
    return x, y, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="gaussian",
                         help="initial bump shape, passed to radial_wavefunction "
                              "(gaussian, quartic, or hydrogenNs); default gaussian")
    parser.add_argument("--width", type=float, default=2.0,
                         help="initial width `a` (default 2.0 -- wide compared to the "
                              "true ground-state width a=1)")
    parser.add_argument("--r0", type=float, default=0.0,
                         help="initial shell radius r0 (default 0.0; >0 for an off-center start)")
    parser.add_argument("--r-max", type=float, default=50.0)
    parser.add_argument("--n-grid", type=int, default=2400)
    parser.add_argument("--t-max", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--max-n", type=int, default=4,
                         help="track population in exact eigenstates n=1..max-n (default 4)")
    parser.add_argument("--cloud-points", type=int, default=3000)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output or f"output/radial_time_evolution_{args.kind}.gif"
    rng = np.random.default_rng(args.seed)

    # --- grid and initial condition ---
    r = np.linspace(0.0, args.r_max, args.n_grid)
    dr = r[1] - r[0]
    r_interior = r[1:-1]

    psi0 = radial_wavefunction(r, args.width, args.kind, r0=args.r0)
    u = np.sqrt(4 * np.pi) * r * psi0
    u = u.astype(complex)
    u_interior = u[1:-1].copy()

    # --- exact eigenstates to project onto, same u = sqrt(4pi)*r*psi convention ---
    n_values = list(range(1, args.max_n + 1))
    eigen_u = {}
    for n in n_values:
        psi_n = radial_wavefunction(r, 1.0, f"hydrogen{n}s", r0=0.0)
        eigen_u[n] = np.sqrt(4 * np.pi) * r * psi_n

    def populations(u_full):
        return {n: np.abs(_trapz(eigen_u[n] * u_full, r)) ** 2 for n in n_values}

    # --- Crank-Nicolson setup (unitary at any dt; factor once, reuse every step) ---
    A, B, V_interior = build_cayley_factors(r_interior, dr, args.dt)
    lu = spla.splu(A)
    V_full = np.empty_like(r)
    V_full[1:-1] = V_interior
    V_full[0] = V_full[-1] = 0.0  # u=0 there exactly; value is never multiplied by anything nonzero

    n_steps = int(round(args.t_max / args.dt))
    frame_steps = np.unique(np.linspace(0, n_steps, args.n_frames).astype(int))

    t_hist = np.empty(len(frame_steps))
    P_hist = np.empty((len(frame_steps), args.n_grid))
    T_hist = np.empty(len(frame_steps))
    V_hist = np.empty(len(frame_steps))
    pop_hist = {n: np.empty(len(frame_steps)) for n in n_values}

    frame_idx = 0
    for step in range(n_steps + 1):
        if frame_idx < len(frame_steps) and step == frame_steps[frame_idx]:
            u_full = np.concatenate([[0.0], u_interior, [0.0]])
            T, Vexp = kinetic_potential(u_full, r, V_full)
            pops = populations(u_full)
            t_hist[frame_idx] = step * args.dt
            P_hist[frame_idx] = np.abs(u_full) ** 2
            T_hist[frame_idx] = T
            V_hist[frame_idx] = Vexp
            for n in n_values:
                pop_hist[n][frame_idx] = pops[n]
            frame_idx += 1
        if step < n_steps:
            u_interior = lu.solve(B @ u_interior)

    E_hist = T_hist + V_hist
    norm_final = _trapz(P_hist[-1], r)
    print(f"norm(t=0)={_trapz(P_hist[0], r):.6f}  norm(t={t_hist[-1]:.1f})={norm_final:.6f}  "
          f"E(t=0)={E_hist[0]:.5f}  E(t={t_hist[-1]:.1f})={E_hist[-1]:.5f}  "
          f"(drift shows discretization error, not physical decay)")
    for n in n_values:
        print(f"  |c_{n}|^2: t=0 -> {pop_hist[n][0]:.5f}   t={t_hist[-1]:.1f} -> {pop_hist[n][-1]:.5f}")

    # --- figure ---
    fig = plt.figure(figsize=(13, 10.5))
    fig.patch.set_facecolor(SURFACE)
    ax_P = fig.add_subplot(2, 2, 1)
    ax_energy = fig.add_subplot(2, 2, 2)
    ax_pop = fig.add_subplot(2, 2, 3)
    ax_cloud = fig.add_subplot(2, 2, 4, projection="3d")
    for ax in (ax_P, ax_energy, ax_pop):
        style_axis(ax)
    style_3d_axis(ax_cloud)
    fig.suptitle(
        f"Real-time evolution: {args.kind} bump, a={args.width:.2f}, r0={args.r0:.2f}  --  "
        "unitary evolution, no relaxation to n=1",
        color=INK_PRIMARY, fontsize=12, y=0.99,
    )

    # --- Panel 1: P(r,t) ---
    window = min(args.r_max, max(3 * args.width + args.r0, 15.0))
    ax_P.set_xlabel("r (Bohr radii)", color=INK_SECONDARY)
    ax_P.set_ylabel("P(r,t) = |u(r,t)|²", color=INK_SECONDARY)
    ax_P.set_title("1. Radial probability density, evolving", color=INK_PRIMARY, fontsize=11)
    ax_P.set_xlim(0, window)
    ax_P.set_ylim(0, P_hist.max() * 1.08)
    (line_P,) = ax_P.plot([], [], color=COLOR_P, linewidth=2)
    ax_P.axvline(0, color=INK_MUTED, linewidth=1, linestyle=":")

    # --- Panel 2: <T>, <V>, E vs t ---
    ax_energy.set_xlim(t_hist.min(), t_hist.max())
    y_lo = min(T_hist.min(), V_hist.min(), E_hist.min())
    y_hi = max(T_hist.max(), V_hist.max(), E_hist.max())
    e_margin = 0.1 * (y_hi - y_lo + 1e-9)
    ax_energy.set_ylim(y_lo - e_margin, y_hi + e_margin)
    ax_energy.set_xlabel("t (atomic units)", color=INK_SECONDARY)
    ax_energy.set_ylabel("energy (Hartree)", color=INK_SECONDARY)
    ax_energy.set_title("2. ⟨T⟩, ⟨V⟩ slosh; E=⟨T⟩+⟨V⟩ stays flat",
                         color=INK_PRIMARY, fontsize=11)
    ax_energy.axhline(0, color=BASELINE, linewidth=1)
    ax_energy.plot(t_hist, E_hist, color=COLOR_E, linewidth=2, label="E")
    ax_energy.plot(t_hist, T_hist, color=COLOR_T, linewidth=1.5, label="⟨T⟩")
    ax_energy.plot(t_hist, V_hist, color=COLOR_V, linewidth=1.5, label="⟨V⟩")
    ax_energy.legend(loc="center right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    (marker_E,) = ax_energy.plot([], [], "o", color=COLOR_E, markersize=8,
                                  markeredgecolor=SURFACE, markeredgewidth=1.5)

    # --- Panel 3: |c_n(t)|^2 ---
    ax_pop.set_xlim(t_hist.min(), t_hist.max())
    ax_pop.set_ylim(0, 1.05)
    ax_pop.set_xlabel("t (atomic units)", color=INK_SECONDARY)
    ax_pop.set_ylabel("|cₙ(t)|²", color=INK_SECONDARY)
    ax_pop.set_title("3. Population in each exact eigenstate (flat = no relaxation)",
                      color=INK_PRIMARY, fontsize=11)
    pop_markers = {}
    for i, n in enumerate(n_values):
        color = POP_COLORS[i % len(POP_COLORS)]
        ax_pop.plot(t_hist, pop_hist[n], color=color, linewidth=1.5, label=f"n={n}")
        (pop_markers[n],) = ax_pop.plot([], [], "o", color=color, markersize=7,
                                         markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax_pop.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=len(n_values))

    # --- Panel 4: 3D electron cloud ---
    ax_cloud.set_title("4. Electron cloud (breathing)", color=INK_PRIMARY, fontsize=11, y=1.0)
    ax_cloud.set_xlabel("x")
    ax_cloud.set_ylabel("y")
    ax_cloud.set_zlabel("z")
    cloud_half = window
    ax_cloud.set_xlim(-cloud_half, cloud_half)
    ax_cloud.set_ylim(-cloud_half, cloud_half)
    ax_cloud.set_zlim(-cloud_half, cloud_half)
    x0, y0, z0 = sample_cloud_from_density(r, P_hist[0], args.cloud_points, rng)
    cloud = ax_cloud.scatter(x0, y0, z0, color=COLOR_P, alpha=0.45, s=8, linewidths=0)
    ax_cloud.scatter([0], [0], [0], color=COLOR_V, s=40, depthshade=False)

    info_text = fig.text(0.5, 0.012, "", ha="center", va="bottom", color=INK_SECONDARY, fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    def update(frame_idx):
        line_P.set_data(r, P_hist[frame_idx])
        marker_E.set_data([t_hist[frame_idx]], [E_hist[frame_idx]])
        for n in n_values:
            pop_markers[n].set_data([t_hist[frame_idx]], [pop_hist[n][frame_idx]])
        xs, ys, zs = sample_cloud_from_density(r, P_hist[frame_idx], args.cloud_points, rng)
        cloud._offsets3d = (xs, ys, zs)
        info_text.set_text(
            f"t = {t_hist[frame_idx]:6.2f}    ⟨T⟩ = {T_hist[frame_idx]:7.3f}    "
            f"⟨V⟩ = {V_hist[frame_idx]:7.3f}    E = {E_hist[frame_idx]:7.3f}"
        )
        return (line_P, marker_E, *pop_markers.values(), cloud, info_text)

    anim = FuncAnimation(fig, update, frames=len(frame_steps), blit=False)
    anim.save(output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
