# CSE220 Online 1 — Predicted Questions + Solutions

**Topic:** Basic signal operations & plotting with NumPy / Matplotlib
**Basis:** Extrapolated from 2024 (discrete array ops) and 2025 (sampled continuous-time ops).

---

## Why these predictions

| Year | Sub A | Sub B | Sub C |
|------|-------|-------|-------|
| 2024 | shift `x[n-k]`, scale-down `x[kn]` | reverse, even/odd | scale-up `x[n/k]`, interpolate |
| 2025 | sub-scale `x(t/k)` + interpolate | sinusoid: time-shift vs phase | reversal + even/odd (continuous) |

**Primitives already tested:** shift, scale (both directions), reversal, even/odd, linear interpolation, sinusoid phase/time equivalence.

**Not yet tested but squarely "signal basics" theory (high-probability this year):**
1. **Combined transformation** `x(at + b)` / `x[an - b]` — the natural fusion of shift + scale.
2. **Energy & power** of a signal (+ the even/odd energy-additivity property).
3. **Signal arithmetic** — sum/product of two signals, and derived quantities.
4. **Basic signal synthesis** — impulse, step, ramp, and building a signal from them.
5. **Time super-scaling** `x(kt)` (continuous compression) — the mirror of 2025's sub-scaling.

Below: each predicted question with a full, submission-ready solution.

---

## Prediction 1 — Combined time transformation (HIGHEST likelihood)

> **Likely form (continuous, sampled):** Implement `transform_signal(t, x, a, b)` returning `y(t) = x(a·t + b)`, where `a` is a nonzero real (may be negative → includes reversal) and `b` is a real shift. Values outside the sampled range are 0. Plot `x(t)` and `y(t)` together.
>
> **Likely form (discrete):** Implement `transform_signal(x, a, b)` returning `y[n] = x[a·n − b]` for the `-8..8` window.

This is the single most probable question because it *combines* every primitive you've been drilled on (shift, scale, reversal) into one operation — the standard "next step" after teaching them separately.

### Continuous solution

```python
import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX, N = -4.0, 4.0, 4001

def x_of_t(t):
    return np.sin(2*np.pi*0.5*t) + 0.5*np.sin(2*np.pi*1.5*t)

def transform_signal(t, x, a, b):
    """
    y(t) = x(a*t + b).
    a < 0 folds (reversal), |a|>1 compresses, |a|<1 stretches, b shifts.
    np.interp handles the resampling; left/right=0 zeroes out-of-range.
    """
    t_query = a * t + b
    return np.interp(t_query, t, x, left=0.0, right=0.0)

def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)
    a, b = 2.0, 1.0                 # example: y(t) = x(2t + 1)
    y = transform_signal(t, x, a, b)

    plt.figure()
    plt.plot(t, x, label="x(t)")
    plt.plot(t, y, label=f"y(t) = x({a}t + {b})")
    plt.title("Combined Time Transformation")
    plt.xlabel("t"); plt.ylabel("Amplitude")
    plt.grid(True); plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

### Discrete solution

```python
import numpy as np
INF = 8

def transform_signal_discrete(x, a, b):
    """y[n] = x[a*n - b] for n in [-8, 8]; a,b integers, a != 0."""
    y = np.zeros_like(x)
    n = np.arange(-INF, INF + 1)
    src = a * n - b                      # source index in original time
    valid = (src >= -INF) & (src <= INF)
    y[n[valid] + INF] = x[src[valid] + INF]
    return y
```

**Explanation to give the examiner:** you evaluate the *output* time axis, map each output sample back to the time it reads from the input (`a·t + b` / `a·n − b`), then fetch (interpolate) that value. Sign of `a` gives reversal; magnitude gives scaling; `b` gives shift. Order-of-operations debate disappears because you compute the composite argument directly.

---

## Prediction 2 — Signal energy & power (+ even/odd energy check)

> **Likely form:** Implement `signal_energy(x)` and `signal_power(x)`. Then verify numerically that `E{x} = E{x_even} + E{x_odd}` using your even/odd decomposition.

Energy/power is core "signal properties" theory that hasn't appeared yet, and the even/odd energy-additivity property is a beloved demonstration question (ties directly into 2024/2025's even-odd tasks).

### Solution

```python
import numpy as np
INF = 8

def signal_energy(x):
    """E = sum |x[n]|^2 over the window."""
    return float(np.sum(np.abs(x)**2))

def signal_power(x):
    """Average power over the finite window (N = 2*INF+1 samples)."""
    return float(np.mean(np.abs(x)**2))

def time_reverse(x):
    return x[::-1]

def even_odd(x):
    xr = time_reverse(x)
    return (x + xr) / 2, (x - xr) / 2      # (even, odd)

def main():
    x = np.zeros(2*INF + 1)
    x[INF]   = 1; x[INF+1] = 0.5; x[INF-1] = 2
    x[INF+2] = 1; x[INF-2] = 0.5

    xe, xo = even_odd(x)
    E, Ee, Eo = signal_energy(x), signal_energy(xe), signal_energy(xo)
    print(f"E(x)={E:.4f}, E(xe)+E(xo)={Ee+Eo:.4f}")   # should match

if __name__ == "__main__":
    main()
```

**Key property to explain:** the cross-term `Σ 2·xe[n]·xo[n]` sums to zero because `xe` is even and `xo` is odd, so their product is odd and cancels over a symmetric window. Hence `E{x} = E{xe} + E{xo}`. For the *continuous* variant, replace the sum with a numerical integral: `np.trapz(x**2, t)`.

---

## Prediction 3 — Signal arithmetic (sum / product of two signals)

> **Likely form:** Given two signals `x1`, `x2` on the same axis, implement `add_signals`, `multiply_signals`, and report the energy of each result. Or: build `y = a1·x1[n-k1] + a2·x2[n-k2]` (a shifted/scaled linear combination).

A natural, low-effort question the graders can dress up with amplitude scaling + shifts, exercising the shift primitive again.

### Solution

```python
import numpy as np
INF = 8

def time_shift(x, k):
    """x[n-k]: k>0 shifts right."""
    y = np.zeros_like(x)
    if k > 0:   y[k:]  = x[:-k]
    elif k < 0: y[:k]  = x[-k:]
    else:       y[:]   = x
    return y

def add_signals(x1, x2):      return x1 + x2
def multiply_signals(x1, x2): return x1 * x2

def linear_combination(x1, x2, a1, a2, k1, k2):
    """y[n] = a1*x1[n-k1] + a2*x2[n-k2]."""
    return a1*time_shift(x1, k1) + a2*time_shift(x2, k2)
```

**Note for continuous version:** signals sampled on the *same* `t` axis add/multiply elementwise directly; if axes differ, resample one onto the other with `np.interp` first.

---

## Prediction 4 — Basic signal synthesis (impulse / step / ramp)

> **Likely form:** Implement `unit_impulse(n0)`, `unit_step(n0)`, `unit_ramp()` on the `-8..8` grid, then construct a target signal as a sum of shifted/scaled versions and plot it.

Generating elementary signals is a textbook "lab 1" task; it fits "basic signals" perfectly and hasn't shown up.

### Solution

```python
import numpy as np
INF = 8
n = np.arange(-INF, INF + 1)

def unit_impulse(n0=0):
    """delta[n - n0]."""
    y = np.zeros_like(n, dtype=float)
    if -INF <= n0 <= INF:
        y[n0 + INF] = 1.0
    return y

def unit_step(n0=0):
    """u[n - n0]."""
    return (n >= n0).astype(float)

def unit_ramp(n0=0):
    """r[n - n0] = (n-n0) for n>=n0 else 0."""
    return np.where(n >= n0, n - n0, 0.0).astype(float)

# Example: build x[n] = 2*delta[n] + delta[n-1] - u[n-3]
x = 2*unit_impulse(0) + unit_impulse(1) - unit_step(3)
```

**Vectorized (bonus-mark) note:** all three use boolean masks / indexing, no Python loops — that's what wins the "use NumPy functions" bonus that appears every year.

---

## Prediction 5 — Time super-scaling `x(kt)` (continuous, mirror of 2025-A)

> **Likely form:** Implement `time_super_scale(t, x, k)` giving `y(t) = x(k·t)` (compression). Discuss which samples survive and whether interpolation is needed.

2025 tested sub-scaling `x(t/k)` (stretch → needs interpolation). The symmetric partner — compression `x(kt)` — is the obvious variant to hand a different subsection.

### Solution

```python
import numpy as np

def time_super_scale(t, x, k):
    """y(t) = x(k*t); compression by factor k. Out-of-range -> 0."""
    return np.interp(k * t, t, x, left=0.0, right=0.0)
```

**Explanation:** compression reads the input at *farther-out* times `k·t`, so no new interior samples are invented — but samples beyond the original range fall off (set to 0). Interpolation is only strictly needed when `k·t` lands between grid points, which `np.interp` handles cleanly. Contrast with sub-scaling, where the *stretch* creates genuine gaps that must be filled by averaging neighbors.

---

## Prediction 6 — Recurring safe bet: reversal + even/odd (could simply repeat)

Reversal and even/odd have appeared **both** years (2024 discrete, 2025 continuous). Keep the one-liners ready; a subsection may just reuse them verbatim.

```python
# discrete
def time_reverse_signal(x):
    return x[::-1]

def odd_even_decomposition(x):
    xr = x[::-1]
    return (x - xr)/2, (x + xr)/2      # (odd, even) — mind the return order!

# continuous (non-uniform / asymmetric safe)
def time_reverse(t, x):
    return np.interp(-t, t, x, left=0.0, right=0.0)
```

> ⚠️ **Return-order trap:** the 2024 spec asks for **(odd, even)** in that order, but it's easy to return `(even, odd)`. Read the spec's return order every time.

---

## Rapid theory cheat-sheet (memorize before the exam)

| Operation | Discrete | Continuous (sampled) | Needs interpolation? |
|-----------|----------|----------------------|----------------------|
| Shift `x[n-k]` / `x(t-b)` | array slice | shift the query time | no / at grid: no |
| Reversal `x[-n]` / `x(-t)` | `x[::-1]` | `np.interp(-t, t, x)` | continuous: yes |
| Down-scale `x[kn]` / `x(kt)` | pick every k-th | `interp(k*t,...)` | continuous: sometimes |
| Up-scale `x[n/k]` / `x(t/k)` | zeros between | avg of neighbors | **yes** |
| Even part | `(x + x_rev)/2` | same | — |
| Odd part | `(x − x_rev)/2` | same | — |
| Energy | `Σ|x|²` | `∫|x|² dt` → `np.trapz(x**2, t)` | — |
| Power | `mean(|x|²)` over window | time-average of `|x|²` | — |

**Recurring gotchas / bonus tips**
- `INF = 8`, array length `2*INF+1 = 17`, array index `= n + INF`.
- Out-of-range → 0 (`left=0.0, right=0.0` in `np.interp`; masks in discrete).
- Vectorize everything (masks, slicing, `np.interp`, `np.where`) — no `for` loops → bonus marks every year.
- Always add `plt.title/xlabel/ylabel/grid/legend`; plotting carries 2–3 marks on its own.
- Check the exact **return order** the spec demands (odd/even ambiguity).
- `np.interp` requires the reference `t` to be **increasing** — it is here, but remember it if axes get flipped.
```

**Single most useful 30 minutes of prep:** implement Prediction 1 (combined `x(at+b)`) cold, since it subsumes shift, scale, and reversal in one function — if you can write that from memory, you can adapt it to almost any variant they throw at you.
