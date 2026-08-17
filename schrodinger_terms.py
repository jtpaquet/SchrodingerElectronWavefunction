"""
Kinetic vs. Coulomb-potential energy terms of a 1D trial wavefunction.

Atomic units throughout: hbar = m_e = e^2/(4*pi*eps0) = 1.

The time-independent Schrodinger equation is split into its two energy
terms:

    H psi = T psi + V psi = -1/2 psi'' + V(x) psi

For a normalized trial wavefunction psi(x), the expectation values are

    <T> = 1/2 * Integral |psi'(x)|^2 dx      (integration by parts,
                                                psi -> 0 at +/-infinity)
    <V> = Integral |psi(x)|^2 V(x) dx

This module builds two families of trial wavefunctions, each shiftable
to a center x0:

  * "hydrogen1s": psi(x; a, x0) = exp(-|x-x0|/a), the 1D analogue of the
    hydrogen 1s radial profile (a is the analogue of the Bohr radius).
  * "gaussian":   psi(x; a, x0) = exp(-(x-x0)^2 / (2 a^2))

against a soft-core 1D Coulomb potential V(x) = -1/sqrt(x^2 + eps^2)
centered on a fixed nucleus at the origin. The softening (eps) removes
the 1/x divergence at the origin so the potential-energy integral stays
finite on a discrete grid; physically it stands in for a nucleus of
finite size.

Two independent sweeps are supported:

  * sweep_widths    -- vary the width `a` with the wavefunction centered
                        on the nucleus (x0 = 0). Models "how spread out
                        is the electron".
  * sweep_positions -- fix the width `a` (a somewhat localized electron)
                        and vary the center x0. Models "how far from the
                        nucleus is the (equally-localized) electron".
                        Note: `x0` is used rather than `r0` (too easily
                        read as a Bohr radius) or `a` (already the width).
"""

import numpy as np

# numpy >=2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

EPS_SOFTENING = 0.2


def make_grid(L=40.0, N=4000):
    return np.linspace(-L, L, N)


def potential(x, eps=EPS_SOFTENING):
    return -1.0 / np.sqrt(x**2 + eps**2)


def wavefunction(x, a, kind="hydrogen1s", x0=0.0):
    dx = x - x0
    if kind == "hydrogen1s":
        psi = np.exp(-np.abs(dx) / a)
    elif kind == "gaussian":
        psi = np.exp(-(dx**2) / (2 * a**2))
    else:
        raise ValueError(f"unknown kind: {kind}")
    norm = np.sqrt(_trapz(psi**2, x))
    return psi / norm


def energy_terms(x, psi, V):
    """Return (T, Vexp, kinetic_density, potential_density)."""
    dpsi_dx = np.gradient(psi, x)
    kinetic_density = 0.5 * dpsi_dx**2
    potential_density = psi**2 * V
    T = _trapz(kinetic_density, x)
    Vexp = _trapz(potential_density, x)
    return T, Vexp, kinetic_density, potential_density


def sweep_widths(a_values, x, V, kind="hydrogen1s"):
    """Precompute T(a), V(a), E(a) with the wavefunction centered at x0=0."""
    T_arr = np.empty_like(a_values)
    V_arr = np.empty_like(a_values)
    for i, a in enumerate(a_values):
        psi = wavefunction(x, a, kind, x0=0.0)
        T, Vexp, _, _ = energy_terms(x, psi, V)
        T_arr[i] = T
        V_arr[i] = Vexp
    E_arr = T_arr + V_arr
    return T_arr, V_arr, E_arr


def sweep_positions(x0_values, x, V, a, kind="gaussian"):
    """Precompute T(x0), V(x0), E(x0) for a fixed-width wavefunction whose
    center x0 is swept away from the nucleus at the origin."""
    T_arr = np.empty_like(x0_values)
    V_arr = np.empty_like(x0_values)
    for i, x0 in enumerate(x0_values):
        psi = wavefunction(x, a, kind, x0=x0)
        T, Vexp, _, _ = energy_terms(x, psi, V)
        T_arr[i] = T
        V_arr[i] = Vexp
    E_arr = T_arr + V_arr
    return T_arr, V_arr, E_arr


def reparam_by_energy_variation(param_fine, T_fine, V_fine, n_samples):
    """Pick `n_samples` values out of the monotonic `param_fine` grid,
    spaced so that each consecutive pair changes <T>+<V> combined by
    about the same amount -- i.e. denser sampling where the terms vary
    fastest, sparser where they're flat.

    Uses |dT|+|dV| rather than |d(T+V)| so the two terms are each kept
    smooth individually: near an energy minimum dE/da=0 by definition,
    but T and V are still changing there (just canceling), and it's
    exactly that region a naive E-only weighting would under-sample.
    """
    weight = np.abs(np.diff(T_fine)) + np.abs(np.diff(V_fine))
    if weight.sum() == 0:
        weight = np.ones_like(weight)
    s = np.concatenate([[0.0], np.cumsum(weight)])
    s /= s[-1]
    target_s = np.linspace(0.0, 1.0, n_samples)
    return np.interp(target_s, s, param_fine)
