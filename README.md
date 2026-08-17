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

## Requirements

```
pip install -r requirements.txt
```
