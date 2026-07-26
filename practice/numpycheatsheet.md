# NumPy Cheat Sheet — DSP Sessional

---

## Array Creation

```python
np.zeros(N)                  # [0, 0, ..., 0]
np.zeros_like(x)             # same shape/dtype as x, filled with 0
np.ones(N)                   # [1, 1, ..., 1]
np.arange(start, stop, step) # [start, start+step, ...) — stop excluded
np.linspace(start, stop, N)  # N evenly spaced points, stop INCLUDED
np.full(N, val)              # array of N copies of val
```

---

## Indexing & Slicing

```python
x[i]                         # single element at index i
x[a:b]                       # slice from a to b (b excluded)
x[::-1]                      # reverse entire array
x[[1, 3, 5]]                 # fancy indexing — pick indices 1,3,5
x[mask]                      # boolean indexing — pick where mask is True
```

---

## Boolean Masks

```python
mask = (x > 0)               # element-wise comparison → bool array
mask = (x >= a) & (x <= b)   # AND  — both conditions
mask = (x < a) | (x > b)     # OR   — either condition
mask = ~mask                  # NOT  — invert mask
np.any(mask)                  # True if at least one True
np.all(mask)                  # True if all True
np.where(mask, a, b)          # a where True, b where False
x[mask]                       # extract elements where True
x[mask] = val                 # assign val where True
```

---

## Arithmetic

```python
x + y  / x - y / x * y / x / y   # element-wise
x ** 2                             # element-wise square
x + c                              # broadcast scalar to all elements
np.abs(x)                          # |x|
np.sqrt(x)                         # √x
np.floor(x).astype(int)            # round down → int
np.ceil(x).astype(int)             # round up  → int
np.mod(x, k)  or  x % k           # remainder — used for upscale valid check
```

---

## Reductions

```python
np.sum(x)                    # Σ x[n]
np.sum(x ** 2)               # energy (discrete)
np.mean(x ** 2)              # power  (discrete)
np.max(x) / np.min(x)        # max / min value
np.argmax(x)                 # index of max value
np.cumsum(x)                 # running sum
```

---

## Signal-Specific

```python
x[::-1]                          # time reversal x[-n]  (symmetric range)
np.roll(x, k)                    # circular shift by k
np.convolve(x, h, mode='full')   # convolution (full output)
np.convolve(x, h, mode='same')   # same length as x
np.interp(t_query, t, x,
          left=0, right=0)       # resample/interpolate x at t_query points
                                 # left/right = value outside range
np.trapz(x ** 2, t)             # ∫ x²(t) dt  — continuous energy
```

---

## Shape & Structure

```python
x.shape                      # tuple of dimensions
x.size                       # total number of elements
len(x)                       # length along first axis
x.reshape(r, c)              # reshape without copying data
x.flatten()                  # collapse to 1D
np.concatenate([a, b])       # join arrays along axis
np.zeros(out_end - out_start + 1)   # size from index range
```

---

## Type Casting

```python
x.astype(int)                # convert float → int (truncates, not rounds)
x.astype(float)              # convert int → float
float(x)                     # scalar array → Python float
```

---

## Common Patterns in DSP Code

```python
# Map output indices back to input array positions
input_indices  = n_input - n_start        # shift so index 0 = n_start

# Valid range check before indexing
valid = (n_in >= n_start) & (n_in <= n_end)
y[n_out[valid] + INF] = x[n_in[valid] + INF]

# Pick every k-th sample (downscale)
is_multiple = (n_out % k == 0)

# Reconstruct time axis from globals
t = np.linspace(T_MIN, T_MAX, (T_MAX - T_MIN) * FS + 1)

# Even / Odd decomposition
x_rev  = x[::-1]
even   = (x + x_rev) / 2
odd    = (x - x_rev) / 2
```

---

## Quick Reference — interp vs fancy indexing

| Situation | Use |
|---|---|
| Continuous / fractional indices | `np.interp` |
| Discrete / integer indices only | fancy indexing `x[arr]` |
| Need 0 outside range | `np.interp(..., left=0, right=0)` |
| Need to write into output | fancy indexing with mask |