"""
Kinetic vs. Coulomb-potential energy terms of a 3D hydrogen-atom trial
wavefunction, restricted to l=0 (spherically symmetric, no theta/phi
dependence).

Atomic units throughout: hbar = m_e = e^2/(4*pi*eps0) = 1. The
time-independent Schrodinger equation splits into its two energy terms:

    H psi = T psi + V psi = -1/2 * Laplacian(psi) + V(r) psi

For a function of r alone, the 3D integrals collapse to 1D integrals
over r with the spherical volume element 4*pi*r^2*dr:

    1    =  4*pi * Integral psi(r)^2 r^2 dr
    <T>  =  2*pi * Integral psi'(r)^2 r^2 dr
    <V>  = -4*pi * Integral psi(r)^2 r dr        (V(r) = -1/r)

The r^2 measure makes -1/r integrable at r=0 on its own -- no soft-core
regularization needed. <T>'s r^2 weight also makes psi'(0) harmless even
though it multiplies a 1/r^2-type curvature term in the full Laplacian;
that term is exactly what integration by parts turns into r^2*psi'^2
here (see README).

Two trial wavefunction shapes, each shiftable to a center r0:

  * "hydrogen1s": psi(r; a, r0) = exp(-|r-r0|/a), the exact hydrogen
    1s radial profile when r0=0 (a is the Bohr radius).
  * "gaussian":   psi(r; a, r0) = exp(-(r-r0)^2 / (2 a^2))

Two independent sweeps are supported:

  * sweep_radial_widths -- the bump is centered on the nucleus (r0=0);
    its width `a` is swept. Models "how spread out is the electron".
  * sweep_radial_shells -- width `a` fixed, and the bump's center r0
    (the radius of a spherical shell of fixed thickness) is swept
    outward. There's no way to "translate" a spherically symmetric
    function off-center without breaking the symmetry, so this -- not
    a Cartesian shift -- is the natural equivalent of moving a
    localized electron away from the nucleus while keeping it equally
    localized.
"""

import numpy as np
from scipy.special import eval_genlaguerre

# numpy >=2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


def reparam_by_energy_variation(param_fine, T_fine, V_fine, n_samples, log_compress=True):
    """Pick `n_samples` values out of the monotonic `param_fine` grid,
    spaced so that each consecutive pair changes <T> and <V> by about
    the same amount -- i.e. denser sampling where the terms vary
    fastest, sparser where they're flat.

    Uses |dT|+|dV| rather than |d(T+V)| so the two terms are each kept
    smooth individually: near an energy minimum dE/da=0 by definition,
    but T and V are still changing there (just canceling), and it's
    exactly that region a naive E-only weighting would under-sample.

    `log_compress` weights by |d(ln T)|+|d(ln|V|)| instead of the raw
    |dT|+|dV|. Near a power-law region (e.g. T~1/a^2 for a small width
    sweep) the raw derivative spans orders of magnitude across the
    sweep, so nearly the entire sample budget piles up at one extreme
    and everywhere else is starved. The log derivative of a pure power
    law is *constant*, so log-compression spends frames proportionally
    to how many e-folds a region covers rather than its raw slope --
    extra density still goes to genuine curvature (e.g. near the energy
    minimum, where the power law breaks down), without starving the
    rest of the range.
    """
    if log_compress:
        wT = np.abs(np.diff(np.log(np.maximum(T_fine, 1e-12))))
        wV = np.abs(np.diff(np.log(np.maximum(np.abs(V_fine), 1e-12))))
        weight = wT + wV
    else:
        weight = np.abs(np.diff(T_fine)) + np.abs(np.diff(V_fine))
    if weight.sum() == 0:
        weight = np.ones_like(weight)
    s = np.concatenate([[0.0], np.cumsum(weight)])
    s /= s[-1]
    target_s = np.linspace(0.0, 1.0, n_samples)
    return np.interp(target_s, s, param_fine)


def make_radial_grid(r_max=25.0, N=20000):
    return np.linspace(0.0, r_max, N)


#: kinds whose shape crosses zero (has a radial node) -- these need a
#: symmetric display range and abs-max normalization in the animation,
#: instead of the [0, max] convention used for a strictly-positive bump.
HAS_NODE = {
    "hydrogen1s": False, "gaussian": False,
    "hydrogen2s": True, "hydrogen3s": True, "hydrogen4s": True,
}

#: exact hydrogen ns (l=0) principal quantum numbers, for the
#: "hydrogenNs" kinds below.
_HYDROGEN_NS_N = {"hydrogen1s": 1, "hydrogen2s": 2, "hydrogen3s": 3, "hydrogen4s": 4}


def radial_wavefunction(r, a, kind="hydrogen1s", r0=0.0):
    dr = r - r0
    if kind == "gaussian":
        psi = np.exp(-(dr**2) / (2 * a**2))
    elif kind in _HYDROGEN_NS_N:
        # The exact hydrogen ns (l=0) shape, rescaled as a whole by `a`
        # (a=1 reproduces the true ns state exactly): psi(r) = e^{-x} *
        # L_{n-1}^1(2x), x = |r-r0|/(n*a), the standard hydrogen radial
        # wavefunction with the associated Laguerre polynomial L_{n-1}^1
        # (verified numerically against the exact energies -1/(2n^2) and
        # exact virial ratio 0.5 for n=1..4 -- see README). n=1 reduces
        # to the plain exponential (L_0^1 is a nonzero constant); r0
        # shifts the whole shape's origin, same as the other kinds, but
        # the natural use here is r0=0 -- see README.
        n = _HYDROGEN_NS_N[kind]
        x = np.abs(dr) / (n * a)
        psi = np.exp(-x) * eval_genlaguerre(n - 1, 1, 2 * x)
    else:
        raise ValueError(f"unknown kind: {kind}")
    norm = np.sqrt(4 * np.pi * _trapz(r**2 * psi**2, r))
    return psi / norm


def radial_energy_terms(r, psi):
    """Return (T, Vexp, kinetic_density, potential_density), all already
    including the 4*pi*r^2 (or 4*pi*r) spherical measure, so a straight
    trapz over r gives the expectation value with no extra factors."""
    dpsi_dr = np.gradient(psi, r)
    kinetic_density = 2 * np.pi * r**2 * dpsi_dr**2
    potential_density = -4 * np.pi * r * psi**2
    T = _trapz(kinetic_density, r)
    Vexp = _trapz(potential_density, r)
    return T, Vexp, kinetic_density, potential_density


def sweep_radial_widths(a_values, r, kind="hydrogen1s"):
    """Precompute T(a), V(a), E(a) for a bump centered on the nucleus (r0=0)."""
    T_arr = np.empty_like(a_values)
    V_arr = np.empty_like(a_values)
    for i, a in enumerate(a_values):
        psi = radial_wavefunction(r, a, kind, r0=0.0)
        T, Vexp, _, _ = radial_energy_terms(r, psi)
        T_arr[i] = T
        V_arr[i] = Vexp
    E_arr = T_arr + V_arr
    return T_arr, V_arr, E_arr


def sweep_radial_shells(r0_values, r, a, kind="gaussian"):
    """Precompute T(r0), V(r0), E(r0) for a fixed-thickness shell whose
    radius r0 is swept outward from the nucleus."""
    T_arr = np.empty_like(r0_values)
    V_arr = np.empty_like(r0_values)
    for i, r0 in enumerate(r0_values):
        psi = radial_wavefunction(r, a, kind, r0=r0)
        T, Vexp, _, _ = radial_energy_terms(r, psi)
        T_arr[i] = T
        V_arr[i] = Vexp
    E_arr = T_arr + V_arr
    return T_arr, V_arr, E_arr


def sample_electron_cloud(r, psi, n_points, rng):
    """Sample `n_points` (x, y, z) points from the 3D probability density
    |psi(r)|^2 of a spherically symmetric (l=0) wavefunction, for a
    scatter-plot "electron cloud" rendering. Radius is drawn by inverse-
    CDF sampling of the radial density 4*pi*r^2*psi(r)^2; direction is
    uniform on the sphere (exact for l=0, which has no angular
    dependence)."""
    radial_density = r**2 * psi**2
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (radial_density[1:] + radial_density[:-1]) * np.diff(r))])
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
