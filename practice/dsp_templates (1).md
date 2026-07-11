# DSP Operations — Boilerplate Templates

```python
import numpy as np

INF   = 8                                           # discrete range [-8, 8]
T_MIN, T_MAX, FS = -8, 8, 1000                     # continuous globals
```

---

## Shift — Discrete `x[n-k]`

```python
def shift_discrete(x: np.ndarray, k: int) -> np.ndarray:
    y = np.zeros_like(x)
    n_out = np.arange(-INF, INF + 1)               # output indices
    n_in  = n_out - k                              # x[n-k] → sample x at (n-k)
    valid = (n_in >= -INF) & (n_in <= INF)
    y[n_out[valid] + INF] = x[n_in[valid] + INF]
    return y
```

## Shift — Continuous `x(t-b)`

```python
def shift_continuous(x: np.ndarray, b: float) -> np.ndarray:
    t = np.linspace(T_MIN, T_MAX, (T_MAX - T_MIN) * FS + 1)
    t_shifted = t - b                              # query x at (t-b)
    y = np.interp(t_shifted, t, x, left=0.0, right=0.0)
    return y
```

---

## Reversal — Discrete `x[-n]`

```python
def reverse_discrete(x: np.ndarray) -> np.ndarray:
    return x[::-1]                                 # symmetric range → plain flip
```

## Reversal — Continuous `x(-t)`

```python
def reverse_continuous(x: np.ndarray) -> np.ndarray:
    t = np.linspace(T_MIN, T_MAX, (T_MAX - T_MIN) * FS + 1)
    y = np.interp(-t, t, x, left=0.0, right=0.0)  # query x at (-t)
    return y
```

---

## Down-scale — Discrete `x[kn]`

```python
def downscale_discrete(x: np.ndarray, k: int) -> np.ndarray:
    y = np.zeros_like(x)
    n_out = np.arange(-INF, INF + 1)
    n_in  = n_out * k                              # x[kn] → sample x at (kn)
    valid = (n_in >= -INF) & (n_in <= INF)
    y[n_out[valid] + INF] = x[n_in[valid] + INF]
    return y
```

## Down-scale — Continuous `x(kt)`

```python
def downscale_continuous(x: np.ndarray, k: float) -> np.ndarray:
    t = np.linspace(T_MIN, T_MAX, (T_MAX - T_MIN) * FS + 1)
    y = np.interp(k * t, t, x, left=0.0, right=0.0)  # query x at (kt)
    return y
```

---

## Up-scale — Discrete `x[n/k]` (zeros between)

```python
def upscale_discrete(x: np.ndarray, k: int) -> np.ndarray:
    y = np.zeros_like(x)
    n_out    = np.arange(-INF, INF + 1)
    is_exact = (n_out % k == 0)                    # only multiples of k map back
    n_in     = (n_out[is_exact] // k)
    valid    = (n_in >= -INF) & (n_in <= INF)
    y[n_out[is_exact][valid] + INF] = x[n_in[valid] + INF]
    return y
```

## Up-scale — Continuous `x(t/k)` (avg of neighbors)

```python
def upscale_continuous(x: np.ndarray, k: float) -> np.ndarray:
    t = np.linspace(T_MIN, T_MAX, (T_MAX - T_MIN) * FS + 1)
    y = np.interp(t / k, t, x, left=0.0, right=0.0)  # query x at (t/k)
    return y
```

---

## Even Part — Discrete & Continuous

```python
def even_part(x: np.ndarray) -> np.ndarray:
    x_rev = x[::-1]                                # works for both domains
    return (x + x_rev) / 2
```

---

## Odd Part — Discrete & Continuous

```python
def odd_part(x: np.ndarray) -> np.ndarray:
    x_rev = x[::-1]
    return (x - x_rev) / 2
```

---

## Energy — Discrete

```python
def energy_discrete(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))
```

## Energy — Continuous

```python
def energy_continuous(x: np.ndarray) -> float:
    t = np.linspace(T_MIN, T_MAX, (T_MAX - T_MIN) * FS + 1)
    return float(np.trapz(x ** 2, t))             # ∫ x²(t) dt
```

---

## Power — Discrete

```python
def power_discrete(x: np.ndarray) -> float:
    return float(np.mean(x ** 2))
```

## Power — Continuous

```python
def power_continuous(x: np.ndarray) -> float:
    t = np.linspace(T_MIN, T_MAX, (T_MAX - T_MIN) * FS + 1)
    T = T_MAX - T_MIN
    return float(np.trapz(x ** 2, t) / T)         # (1/T) ∫ x²(t) dt over window
```
