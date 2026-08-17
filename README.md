# Schrödinger electron wavefunction: kinetic vs. potential energy

Visualizes how the two terms of the time-independent Schrödinger equation,

    H psi = T psi + V psi = -1/2 psi''(x) + V(x) psi(x)      (atomic units)

trade off against each other for a 1D trial wavefunction, by computing

    <T> = 1/2 * Integral |psi'(x)|^2 dx
    <V> = Integral |psi(x)|^2 V(x) dx

against a soft-core 1D Coulomb potential `V(x) = -1/sqrt(x^2 + eps^2)`
(a nucleus at the origin; `eps` regularizes the 1/x divergence -- see
"Why soft-core?" below).

Two trial wavefunction shapes are supported (`schrodinger_terms.py`):
`hydrogen1s`: `exp(-|x-x0|/a)`, and `gaussian`: `exp(-(x-x0)^2/2a^2)`.

## Two sweeps, `animate_energy_terms.py`

**`width`** -- the wavefunction stays centered on the nucleus (`x0=0`)
while its width `a` is swept from very large (spread out) down to very
small (localized) and back. Models "how spread out is the electron."

```
python animate_energy_terms.py width --kind hydrogen1s -o output/width_hydrogen1s.gif
python animate_energy_terms.py width --kind gaussian    -o output/width_gaussian.gif
```

**`position`** -- the width `a` is held fixed at a somewhat localized
value (default 1.0) while the wavefunction's center `x0` is swept away
from and back across the nucleus. Models "how far from the nucleus is
the (equally localized) electron." (Named `x0`, not `r0` -- too easily
misread as a Bohr radius -- and not `a`, already the width.)

```
python animate_energy_terms.py position --kind gaussian -o output/position_gaussian.gif
```

Each run renders three panels per frame: the wavefunction, total energy
`E(a)` or `E(x0)` traced across the whole sweep with a marker for the
current frame, and a bar chart comparing `<T>` against `|<V>|` with
their ratio annotated.

## Why soft-core?

In 3D, the hydrogen 1s trial family `psi(r;a) = exp(-r/a)` needs no
regularization: the volume element `r^2 dr` makes `1/r` integrable at
the origin, and `<T>(a) = 1/(2a^2)`, `<V>(a) = -1/a` follow from clean
scaling. Minimizing `E(a) = <T>+<V>` gives `a*=1` (the Bohr radius) and
`E*=-0.5` Hartree -- the exact hydrogen ground state -- with the virial
theorem `2<T> = |<V>|` holding exactly at that minimum, for *any* trial
shape related by pure dilation, not just this one.

In 1D the same integral (`dx`, not `r^2 dr`) diverges at `x=0` for any
wavefunction with `psi(0) != 0`, so `-1/|x|` has to be softened to stay
finite on a grid. That regularization breaks the exact `1/a` scaling of
`<V>(a)`, so the width-sweep's energy minimum lands close to, but not
exactly at, `<T>/|<V>| = 0.5` here -- the deviation shrinks as `eps/a`
shrinks. This is a genuine artifact of collapsing a radial problem onto
a line, not a bug in the integration.

## The genuine 3D model (no soft-core), `animate_radial_energy_terms.py`

Restricting to l=0 (spherically symmetric, no theta/phi dependence)
trial wavefunctions `psi(r)` turns the full 3D integrals into 1D
integrals over `r` with the spherical measure `4*pi*r^2*dr`:

    1    =  4*pi * Integral psi(r)^2 r^2 dr
    <T>  =  2*pi * Integral psi'(r)^2 r^2 dr
    <V>  = -4*pi * Integral psi(r)^2 r dr        (V(r) = -1/r)

The `r^2`/`r` weight makes `-1/r` integrable at `r=0` on its own -- no
softening, and the width sweep reproduces the exact hydrogen ground
state (`a*=1`, `E*=-0.5`, virial ratio exactly `0.5`, to numerical
precision) instead of the 1D model's approximation of it.

```
python animate_radial_energy_terms.py width --kind hydrogen1s -o output/radial_width_hydrogen1s.gif
python animate_radial_energy_terms.py width --kind gaussian    -o output/radial_width_gaussian.gif
python animate_radial_energy_terms.py shell  --kind gaussian    -o output/radial_shell_gaussian.gif
```

Four panels per frame: the radial wavefunction `psi(r)`; `<T>` and
total energy `E` traced across the sweep (log-`x` in `width` mode, so
the `1/a^2` blow-up at small `a` and the flat large-`a` tail are both
visible); a bar chart of `<T>` vs. `|<V>|` with their ratio; and a 3D
"electron cloud" -- points sampled from `|psi(r)|^2` (uniform on the
sphere at each sampled radius, exact for l=0), drawn with low alpha so
density reads visually.

**`shell` mode** replaces the flat 1D model's `position` sweep. You
can't translate a spherically symmetric function off-center without
breaking the symmetry, so the 3D equivalent of "how far from the
nucleus is a localized electron" is the radius `r0` of a spherical
shell of fixed thickness `a`, not a Cartesian shift.

This *can* produce a genuine energy minimum away from the nucleus --
but it's worth being precise about why, since it's a different effect
from the one people usually mean by "the electron doesn't want to sit
on the nucleus." The real hydrogen 1s wavefunction `psi(r) = exp(-r/a)`
is actually *largest* at `r=0`; what peaks at the Bohr radius instead
is the radial probability density `4*pi*r^2*psi(r)^2`, purely from the
`r^2` volume factor (shells at larger `r` have more room), with no
change of shape in `psi` itself required.

The `shell` sweep's off-center minimum is a *different* mechanism: a
Gaussian shell of thickness `a` centered at `r0=0` gets clipped in half
by the `r>=0` boundary (the other half would live at `r<0`, which
doesn't exist), and renormalizing that clipped half makes it taller and
steeper -- i.e. costs kinetic energy. As `r0` grows past roughly `1.5*a`
the shell clears the boundary and that penalty vanishes, while `<V>`
only weakens slowly (`~1/r0`). The competition between a fast-vanishing
kinetic penalty and a slow-fading potential benefit is what produces an
interior minimum, and it only shows up for `a` thin enough that the
clipping is significant -- the default `--fixed-width 0.7` is chosen to
show it clearly while staying bound (`E<0`); try `--fixed-width 1.0` to
see it disappear (the shell is fat enough that clipping barely matters,
same qualitative shape as the flat 1D `position` sweep), or
`--fixed-width 0.3` to see it dramatically (a deep, purely repulsive
kinetic penalty near `r0=0`, unbound throughout).

## Requirements

```
pip install -r requirements.txt
```
