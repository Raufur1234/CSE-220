# Sampling Coding Test Guide

This is a copy-first reference for a NumPy/Matplotlib coding test on sampling. The companion file `sampling_skeleton.py` contains tested versions of the reusable helpers.

Start most programs with:

```python
import numpy as np
import matplotlib.pyplot as plt

from sampling_skeleton import *
```

## 1. Syllabus Boundary and How to Use This Guide

Included: the lecture from sampling terminology through the complete zero-order-hold (ZOH) frequency response, PDF pages 1-31.

- Uniform sampling and ambiguity
- Impulse-train sampling
- Multiplication in time / convolution in frequency
- Spectrum replication
- Nyquist-Shannon condition
- Oversampling, undersampling, and aliasing
- Ideal low-pass and sinc reconstruction
- Zero-order hold

Excluded: linear interpolation / first-order hold and the later DFT derivations. This guide uses `np.fft` only as a numerical spectrum-visualization tool, not as coverage of that later theory.

Suggested exam workflow:

1. Write the signal formula and list every frequency in hertz.
2. Set `f_max = max(frequencies)` and check `fs > 2*f_max`.
3. Build a dense time array only for a smooth-looking reference curve.
4. Build the actual sample times from `fs`.
5. Evaluate the same vectorized signal function on both arrays.
6. Plot, label, and state the theoretical conclusion.
7. If reconstruction is requested, select ideal sinc or ZOH explicitly.

## 2. Formula and Notation Sheet

| Meaning | Hertz notation | Angular-frequency notation |
|---|---:|---:|
| Sampling frequency | `fs` samples/s | `omega_s = 2*pi*fs` rad/s |
| Sampling period | `T = 1/fs` s | `omega_s = 2*pi/T` |
| Highest signal frequency | `f_max` Hz | `omega_M = 2*pi*f_max` rad/s |
| Nyquist rate | `2*f_max` | `2*omega_M` |

The lecture uses the strict condition:

```text
fs > 2*f_max
omega_s > 2*omega_M
```

Sampling exactly at equality is fragile. For example, samples of a sine can all land on zero. Unless a question explicitly accepts the textbook shorthand, choose a rate strictly greater than the Nyquist rate.

Impulse-train model:

```text
p(t)  = sum_n delta(t - nT)
xp(t) = x(t)p(t) = sum_n x(nT) delta(t - nT)
```

Frequency-domain copies:

```text
P(j omega)  = (2*pi/T) sum_k delta(omega - k*omega_s)
Xp(j omega) = (1/T) sum_k X(j(omega - k*omega_s))
```

Each copy is spaced by `omega_s` and scaled by `1/T`. If adjacent copies overlap, aliasing has mixed them and an ideal filter cannot unmix them.

Ideal reconstruction:

```text
H(j omega) = T for |omega| < pi/T, otherwise 0
h(t)       = sinc(t/T)
xr(t)      = sum_n x[n] sinc((t - nT)/T)
```

NumPy uses the normalized sinc:

```text
np.sinc(u) = sin(pi*u)/(pi*u)
```

Zero-order hold:

```text
h0(t) = 1 on [0, T), otherwise 0
H0(j omega) = T exp(-j*omega*T/2) sinc(omega*T/(2*pi))
H0(f)       = T exp(-j*pi*f*T) sinc(f*T)
```

The magnitude is `T*abs(sinc(f*T))`. It has DC gain `T`, droops with frequency, and has its first null at `f = fs = 1/T`.

## 3. NumPy Essentials

### 3.1 Array creation

Minimal forms:

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
integers = np.arange(0, 8, 2)          # [0, 2, 4, 6]
grid = np.linspace(0.0, 1.0, 5)       # includes both endpoints
z = np.zeros(4)
o = np.ones(4)
c = np.full(4, 7.0)
same_shape = np.zeros_like(a)
```

Sampling-related forms:

```python
fs = 20.0
T = 1.0 / fs
t_samples = np.arange(0.0, 1.0, T)       # half-open [0, 1)
t_dense = np.linspace(0.0, 1.0, 2001)    # reference plot, includes 1

# More reliable when duration*fs should be an integer:
N = int(1.0 * fs)
t_samples = np.arange(N) / fs
```

`np.arange` is ideal for a known step. `np.linspace(start, stop, count)` is ideal for a known number of points. Do not confuse its third argument with a sample rate.

### 3.2 Inspecting and copying arrays

```python
x = np.array([1, 2, 3], dtype=np.int64)
print(x.shape)   # (3,)
print(x.ndim)    # 1
print(x.size)    # 3
print(x.dtype)   # int64

y = x.copy()             # independent data
y = x.astype(float)      # converted copy
row = x.reshape(1, -1)   # shape (1, 3)
column = x.reshape(-1, 1)  # shape (3, 1)
```

Sampling check:

```python
t = np.arange(8) / 8.0
x = np.sin(2 * np.pi * 1.0 * t)
assert t.shape == x.shape
assert t.ndim == 1
```

### 3.3 Indexing, slicing, and assignment

```python
x = np.array([10, 20, 30, 40, 50])
first = x[0]
last = x[-1]
middle = x[1:4]       # indices 1, 2, 3
every_other = x[::2]
reversed_x = x[::-1]
x[1:3] = 0
```

Sampling-related slicing:

```python
fs = 100
t = np.arange(100) / fs
x = np.cos(2 * np.pi * 5 * t)
first_half = x[:50]
even_numbered_samples = x[::2]
```

Slicing usually returns a view. Use `.copy()` before modifying a slice when the original must remain unchanged.

### 3.4 Masks, Boolean operators, and `np.where`

```python
x = np.array([-2, -1, 0, 1, 2])
positive = x > 0
selected = x[positive]
x_clipped = x.copy()
x_clipped[x_clipped < 0] = 0

inside = (x >= -1) & (x <= 1)
outside = (x < -1) | (x > 1)
not_inside = ~inside
y = np.where(x >= 0, x, -x)
```

Use `&`, `|`, and `~` for array logic, not Python's `and`, `or`, and `not`. Put each comparison in parentheses.

Piecewise signal:

```python
t = np.linspace(-2.0, 2.0, 1001)
x = np.zeros_like(t)
x[(t >= -1.0) & (t < 0.0)] = t[(t >= -1.0) & (t < 0.0)] + 1.0
x[(t >= 0.0) & (t <= 1.0)] = 1.0 - t[(t >= 0.0) & (t <= 1.0)]

# Same triangle with np.where:
x2 = np.where(np.abs(t) <= 1.0, 1.0 - np.abs(t), 0.0)
assert np.allclose(x, x2)
```

### 3.5 Broadcasting and vectorization

```python
column = np.array([1, 2, 3])[:, None]   # (3, 1)
row = np.array([10, 20])[None, :]       # (1, 2)
table = column + row                     # (3, 2)
```

Sampling-related vectorization:

```python
frequencies = np.array([3.0, 7.0, 12.0])
t = np.arange(100) / 100.0

# One row per tone; shape is (3, 100).
tones = np.sin(2 * np.pi * frequencies[:, None] * t[None, :])
x = tones.sum(axis=0)
```

Vectorized array formulas are clearer and faster than loops over every time instant.

### 3.6 Combining, padding, differences, and reductions

```python
a = np.array([1, 2])
b = np.array([3, 4])
joined = np.concatenate([a, b])
rows = np.stack([a, b], axis=0)
padded = np.pad(a, (2, 1), constant_values=0)

dx = np.diff(joined)
total = np.sum(joined)
average = np.mean(joined)
largest = np.max(joined)
largest_index = np.argmax(joined)
close = np.allclose(joined, [1, 2, 3, 4])
```

Sampling uses:

```python
from sampling_skeleton import zero_pad

x = np.array([1.0, -1.0, 0.5])
x_padded = zero_pad(x, left=2, right=3)

t = np.arange(6) / 10.0
T_estimated = np.mean(np.diff(t))
uniform = np.allclose(np.diff(t), T_estimated)
peak_index = np.argmax(np.abs(x))
```

Zero padding adds stored zeros; it does not create new information or increase the actual sampling rate.

### 3.7 Mathematical functions and a numerical spectrum

```python
t = np.linspace(0.0, 1.0, 1001)
sine = np.sin(2 * np.pi * 3 * t)
cosine = np.cos(2 * np.pi * 3 * t)
decay = np.exp(-2 * t)
magnitude = np.abs(decay * np.exp(1j * 2 * np.pi * 3 * t))
sinc_values = np.sinc(t)
```

Spectrum visualization:

```python
from sampling_skeleton import magnitude_spectrum

fs = 100.0
t = np.arange(int(fs)) / fs
x = np.sin(2 * np.pi * 12 * t)
f, amplitude = magnitude_spectrum(x, fs)
print(f[np.argmax(amplitude)])  # 12.0 Hz
```

Equivalent direct NumPy pattern:

```python
X = np.fft.rfft(x)
f = np.fft.rfftfreq(x.size, d=1 / fs)
amplitude = 2 * np.abs(X) / x.size
amplitude[0] /= 2
if x.size % 2 == 0:
    amplitude[-1] /= 2
```

This is a finite-record numerical estimate, not an array of mathematical Dirac impulses and not a substitute for the CTFT derivation.

## 4. Matplotlib Essentials

### 4.1 Line and stem plots

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, x, color="C0", linewidth=2, label="x(t)")
ax.set(title="Signal", xlabel="Time (s)", ylabel="Amplitude")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()
```

Discrete samples:

```python
fig, ax = plt.subplots()
ax.stem(t_samples, x_samples, linefmt="C1-", markerfmt="C1o", basefmt="k-")
ax.set(xlabel="Time (s)", ylabel="x[n]", title="Samples")
ax.grid(True, alpha=0.3)
plt.show()
```

### 4.2 Overlaying reference and samples

```python
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_dense, x_dense, label="dense reference")
ax.stem(t_samples, x_samples, linefmt="C1-", markerfmt="C1o", basefmt="k-")
ax.set_xlim(t_dense[0], t_dense[-1])
ax.set_ylim(-1.2, 1.2)
ax.set(xlabel="Time (s)", ylabel="Amplitude")
ax.legend(["dense reference", "samples"])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

Or use the skeleton:

```python
fig, ax = plot_sampling(t_dense, x_dense, t_samples, x_samples)
plt.show()
```

A dense plotted curve is still a finite array. It only looks continuous on the screen.

### 4.3 Subplots and saving

```python
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(t_dense, x_dense)
axes[0].set_title("Reference")
axes[1].stem(t_samples, x_samples)
axes[1].set_title("Samples")
for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Amplitude")
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("sampling_result.png", dpi=200, bbox_inches="tight")
plt.show()
```

The helper `plot_signal(t, x, discrete=False)` returns `(fig, ax)` without calling `plt.show()`, so it remains easy to customize.

## 5. Building Signals with Arrays and Masks

### 5.1 Sinusoids and sums of tones

```python
def signal(t):
    return 2 * np.cos(2 * np.pi * 5 * t + np.pi / 4) + np.sin(2 * np.pi * 12 * t)
```

Read frequencies only from the arguments multiplying `t` after converting angular frequency if necessary:

```text
cos(20*pi*t) -> 2*pi*f = 20*pi -> f = 10 Hz
sin(14*t)    -> omega = 14 rad/s -> f = 14/(2*pi) Hz
```

For the function above, `f_max = 12 Hz`, so the strict condition is `fs > 24 Hz`.

### 5.2 Unit steps and pulses

```python
t = np.linspace(-2, 2, 1001)
u = unit_step(t)
pulse = rectangular_pulse(t, start=-0.5, stop=0.5)

# A pulse made from two steps:
pulse2 = unit_step(t, at=-0.5) - unit_step(t, at=0.5)
assert np.allclose(pulse, pulse2)
```

The half-open convention `[start, stop)` prevents overlapping boundaries when adjacent piecewise intervals are combined.

### 5.3 Piecewise signals with masks

For

```text
x(t) = 0          t < -1
       t + 1      -1 <= t < 0
       1          0 <= t < 1
       exp(-(t-1)) t >= 1
```

use:

```python
t = np.linspace(-2, 3, 1001)
x = np.zeros_like(t)

m1 = (t >= -1) & (t < 0)
m2 = (t >= 0) & (t < 1)
m3 = t >= 1

x[m1] = t[m1] + 1
x[m2] = 1
x[m3] = np.exp(-(t[m3] - 1))
```

Alternatively:

```python
x = np.piecewise(
    t,
    [t < -1, m1, m2, m3],
    [0, lambda z: z + 1, 1, lambda z: np.exp(-(z - 1))],
)
```

Masks are often easier to debug in a timed test.

## 6. Uniform Sampling Workflow

### 6.1 Standard template

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import make_time_axis, plot_sampling, sample_signal


def x_of_t(t):
    return np.cos(2 * np.pi * 4 * t) + 0.5 * np.sin(2 * np.pi * 9 * t)


fs = 25.0
start, stop = 0.0, 1.0

t_dense = np.linspace(start, stop, 2001)
x_dense = x_of_t(t_dense)
t_samples, x_samples = sample_signal(x_of_t, fs, start, stop)

print("number of samples:", len(t_samples))
print("T:", 1 / fs)
fig, ax = plot_sampling(t_dense, x_dense, t_samples, x_samples)
plt.show()
```

Here `f_max = 9 Hz`, and `25 > 18`, so the theorem permits ideal recovery.

### 6.2 Sample count and endpoints

For a half-open interval of duration `D` with integer `D*fs`:

```text
N = D*fs
t[n] = start + n/fs, n = 0, 1, ..., N-1
```

Example: one second at `fs=10 Hz` gives ten samples at `0.0, 0.1, ..., 0.9`. Including `t=1.0` gives eleven displayed points, but there are still only ten sampling intervals in that one-second duration.

Use `make_time_axis(start, stop, fs, endpoint=True)` only when the endpoint is explicitly required and lies on the grid.

## 7. Sampling Theorem and Aliasing

### 7.1 Choosing a rate

```python
frequencies_hz = np.array([5.0, 12.0, 30.0])
f_max = frequencies_hz.max()
minimum_rate = nyquist_rate(f_max)

print(minimum_rate)                    # 60.0
print(meets_nyquist(60.0, f_max))      # False: strict lecture condition
print(meets_nyquist(61.0, f_max))      # True
```

If a setter asks for the Nyquist rate, answer `60 Hz`. If asked for a sampling rate satisfying the lecture's condition, answer any `fs > 60 Hz`, such as `61 Hz` or a convenient higher rate.

### 7.2 Predicting aliases

For a real sinusoid, fold its frequency into `[0, fs/2]`:

```python
print(alias_frequency(9.0, fs=10.0))   # 1.0
print(alias_frequency(14.0, fs=10.0))  # 4.0
```

Manual method: choose an integer `k` so that

```text
f_alias = |f - k*fs|
```

falls in `[0, fs/2]`.

### 7.3 Same samples, different continuous signals

If two complex-tone frequencies differ by an integer multiple of `fs`, their sample phases differ by an integer multiple of `2*pi*n`:

```python
fs = 10.0
t_samples = np.arange(10) / fs
x1 = np.sin(2 * np.pi * 3 * t_samples)
x2 = np.sin(2 * np.pi * 13 * t_samples)
print(np.allclose(x1, x2))  # True
```

The continuous curves differ between sample locations, showing why samples alone are ambiguous without a band-limit assumption.

### 7.4 Spectrum-copy picture

Sampling produces shifted copies at `k*fs` in hertz, or `k*omega_s` in rad/s. A finite NumPy array cannot store ideal Dirac impulses. For coding questions, plot conceptual copy boundaries or plot a finite-record FFT and explicitly call it an approximation.

Oversampling leaves a guard band between copies. Undersampling overlaps copies and causes aliasing.

## 8. Ideal Sinc Reconstruction

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, sinc_reconstruct


def signal(t):
    return np.sin(2 * np.pi * 3 * t)


fs = 10.0
t_samples, x_samples = sample_signal(signal, fs, -1.0, 1.1)
t_output = np.linspace(-1.0, 1.0, 2001)
x_reconstructed = sinc_reconstruct(t_samples, x_samples, t_output)

plt.plot(t_output, signal(t_output), label="original")
plt.plot(t_output, x_reconstructed, "--", label="finite sinc reconstruction")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Vectorized core:

```python
T = 1 / fs
kernel = np.sinc((t_output[:, None] - t_samples[None, :]) / T)
x_reconstructed = kernel @ x_samples
```

Each column is one shifted sinc basis function. The theoretical sum is infinite. A finite record truncates it, so error is usually largest near the record edges. At the supplied sample locations, normalized sinc interpolation returns the sample values exactly (apart from roundoff).

## 9. Zero-Order Hold

### 9.1 Time-domain reconstruction

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, zero_order_hold


def signal(t):
    return np.sin(2 * np.pi * 2 * t)


fs = 10.0
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.1)
t_output = np.linspace(0.0, 1.0, 2001)
x_zoh = zero_order_hold(t_samples, x_samples, t_output)

plt.plot(t_output, signal(t_output), label="original")
plt.step(t_output, x_zoh, where="post", label="ZOH")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

The helper holds `x[n]` over `[nT, (n+1)T)`. It uses a fill value outside the recorded sample range instead of silently extrapolating.

Without the helper:

```python
indices = np.searchsorted(t_samples, t_output, side="right") - 1
valid = (t_output >= t_samples[0]) & (t_output <= t_samples[-1])
x_zoh = np.full(t_output.shape, np.nan)
x_zoh[valid] = x_samples[indices[valid]]
```

### 9.2 Frequency response and droop

```python
from sampling_skeleton import zoh_frequency_response

T = 0.1
f = np.linspace(-20.0, 20.0, 4001)
H0 = zoh_frequency_response(f, T)

plt.plot(f, np.abs(H0))
plt.axvline(1 / T, color="C1", linestyle="--", label="first null: fs")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H0(f)|")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

print(abs(zoh_frequency_response([0.0], T)[0]))   # 0.1
print(abs(zoh_frequency_response([10.0], T)[0]))  # approximately 0
```

ZOH is practical but not ideal: its passband gain is not flat, and it does not completely remove shifted spectral copies.

## 10. Copy-Ready Problem Templates

### Template A: define, sample, and overlay

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import plot_sampling, sample_signal


def signal(t):
    return np.cos(2 * np.pi * F1 * t) + A2 * np.sin(2 * np.pi * F2 * t)


fs = FS
t_dense = np.linspace(START, STOP, 2001)
t_samples, x_samples = sample_signal(signal, fs, START, STOP)
fig, ax = plot_sampling(t_dense, signal(t_dense), t_samples, x_samples)
plt.show()
```

Replace the uppercase names with numbers.

### Template B: piecewise signal

```python
t = np.linspace(START, STOP, 2001)
x = np.zeros_like(t)

m1 = (t >= A) & (t < B)
m2 = (t >= B) & (t <= C)
x[m1] = FORMULA_1_USING_T[m1]
x[m2] = FORMULA_2_USING_T[m2]

plot_signal(t, x)
plt.show()
```

### Template C: spectrum visualization

```python
t_samples, x_samples = sample_signal(signal, fs, START, STOP)
f, amplitude = magnitude_spectrum(x_samples, fs, remove_mean=True)
plt.stem(f, amplitude)
plt.xlim(0, fs / 2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Estimated amplitude")
plt.grid(alpha=0.3)
plt.show()
```

### Template D: ideal sinc reconstruction

```python
t_output = np.linspace(t_samples[0], t_samples[-1], 2001)
x_sinc = sinc_reconstruct(t_samples, x_samples, t_output)
plt.plot(t_output, x_sinc, label="sinc reconstruction")
plt.stem(t_samples, x_samples)
plt.legend()
plt.show()
```

### Template E: ZOH reconstruction

```python
t_output = np.linspace(t_samples[0], t_samples[-1], 2001)
x_zoh = zero_order_hold(t_samples, x_samples, t_output)
plt.step(t_output, x_zoh, where="post", label="ZOH")
plt.stem(t_samples, x_samples)
plt.legend()
plt.show()
```

## 11. Common Mistakes and Debugging Checklist

- **Mixing Hz and rad/s:** `cos(omega*t)` has `f = omega/(2*pi)` Hz. `cos(2*pi*f*t)` already uses hertz.
- **Using the Nyquist rate as if strict equality were safe:** the lecture condition is `fs > 2*f_max`.
- **Calling the dense plot continuous:** it is a high-resolution numerical reference.
- **Using `linspace(..., fs)` for one second without thinking:** the third argument is number of points, not samples per second.
- **Including both endpoints accidentally:** `np.linspace(0, 1, 11)` has 11 points but ten intervals.
- **Writing `and` for masks:** use `(condition1) & (condition2)`.
- **Forgetting elementwise operations:** arrays use `*`, `/`, and `**` elementwise; do not use `math.sin` on arrays.
- **Looping unnecessarily:** define signal functions with NumPy operations.
- **Treating an FFT bin as a Dirac impulse:** it is a finite-record numerical value.
- **Forgetting finite-sinc edge error:** the exact theory assumes infinitely many samples.
- **Extending ZOH outside recorded data silently:** decide and state a fill or extrapolation rule.
- **Calling zero padding oversampling:** padding changes array length, not the data-acquisition rate.

Quick diagnostics:

```python
print(t.shape, x.shape)
print("estimated T:", np.mean(np.diff(t)))
print("estimated fs:", 1 / np.mean(np.diff(t)))
print("uniform:", np.allclose(np.diff(t), np.diff(t)[0]))
print("finite:", np.all(np.isfinite(x)))
```

## 12. Practice Problems

Try these before opening the solution section.

### Problem 1: Arrays, slices, and masks

Create `n = [0, 1, ..., 11]`. Form `x[n] = n^2 - 5n`. Print the even-indexed samples, replace every negative value with zero using a mask, and zero-pad the result with two zeros on each side.

### Problem 2: Piecewise signal

On `-2 <= t <= 2`, construct

```text
x(t) = 1 - |t|, |t| <= 1
       0,       otherwise
```

using masks or `np.where`. Plot it with correct labels and limits.

### Problem 3: Safe sinusoid sampling

Let `x(t) = cos(2*pi*6*t)`. Sample it over `[0, 1)` at `fs=20 Hz`. Plot the samples over a dense reference and state whether the strict Nyquist condition holds.

### Problem 4: Rate for a multitone signal

For

```text
x(t) = 2cos(10*pi*t) + sin(60*pi*t) - 0.5cos(24*pi*t),
```

list the component frequencies, find the Nyquist rate, and choose an integer sampling rate satisfying the lecture's strict condition. Generate one second of samples.

### Problem 5: Predict and display aliasing

Sample `x(t) = sin(2*pi*9*t)` at `fs=10 Hz` for one second. Predict the nonnegative alias frequency, then compare the samples with a `1 Hz` sinusoid. Account for any sign difference.

### Problem 6: Same samples from different signals

At `fs=10 Hz`, verify numerically that `sin(2*pi*3*t)` and `sin(2*pi*13*t)` have equal samples over `[0, 1)`. Plot both dense curves and the shared samples.

### Problem 7: Ideal sinc reconstruction

Sample `sin(2*pi*3*t)` at `fs=10 Hz` from `t=-1` through `t=1` and reconstruct it on a dense grid using sinc interpolation. Verify exact agreement at sample positions and comment on the error near the record edges.

### Problem 8: Zero-order hold

Sample `cos(2*pi*2*t)` at `fs=10 Hz` over `[0, 1]`. Reconstruct it on a dense grid with ZOH and plot original, samples, and held result.

### Problem 9: ZOH frequency response

For `T=0.1 s`, compute and plot `|H0(f)|` from `-20 Hz` to `20 Hz`. Report its DC gain and first positive null.

### Problem 10: Mixed sessional problem

Let

```text
x(t) = sin(2*pi*4*t) + 0.4cos(2*pi*11*t).
```

Use `fs=30 Hz` on `[0, 1)`. Confirm Nyquist, create the samples, select samples with positive amplitude using a mask, pad the sample array with four zeros on the right, compute a numerical magnitude spectrum, and compare sinc and ZOH reconstructions on one figure.

## 13. Complete Solutions

### Solution 1

```python
import numpy as np
from sampling_skeleton import zero_pad

n = np.arange(12)
x = n**2 - 5*n
print("even indices:", x[::2])

x_nonnegative = x.copy()
x_nonnegative[x_nonnegative < 0] = 0
x_padded = zero_pad(x_nonnegative, left=2, right=2)
print(x_padded)
```

The mask modifies all negative entries at once. Padding creates a longer stored sequence but does not change a sampling rate.

### Solution 2

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-2.0, 2.0, 1001)
x = np.where(np.abs(t) <= 1.0, 1.0 - np.abs(t), 0.0)

plt.plot(t, x)
plt.xlim(-2, 2)
plt.ylim(-0.1, 1.1)
plt.xlabel("Time (s)")
plt.ylabel("x(t)")
plt.title("Triangular pulse")
plt.grid(alpha=0.3)
plt.show()
```

The condition produces one vectorized Boolean mask; `np.where` chooses the inside and outside formulas elementwise.

### Solution 3

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import meets_nyquist, plot_sampling, sample_signal


def signal(t):
    return np.cos(2 * np.pi * 6 * t)


fs = 20.0
t_dense = np.linspace(0.0, 1.0, 2001)
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0)
print(meets_nyquist(fs, 6.0))  # True because 20 > 12
plot_sampling(t_dense, signal(t_dense), t_samples, x_samples)
plt.show()
```

The strict Nyquist condition holds, so ideal recovery is possible under the band-limited model.

### Solution 4

```python
import numpy as np
from sampling_skeleton import nyquist_rate, sample_signal


def signal(t):
    return 2*np.cos(10*np.pi*t) + np.sin(60*np.pi*t) - 0.5*np.cos(24*np.pi*t)


frequencies = np.array([5.0, 30.0, 12.0])
f_max = frequencies.max()
print("Nyquist rate:", nyquist_rate(f_max))  # 60 Hz

fs = 61.0
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0)
print(len(x_samples))  # 61
```

The maximum component is `30 Hz`, so the Nyquist rate is `60 Hz`; the strict lecture condition requires `fs > 60 Hz`.

### Solution 5

```python
import numpy as np
from sampling_skeleton import alias_frequency

fs = 10.0
t = np.arange(10) / fs
x9 = np.sin(2 * np.pi * 9 * t)
x1 = np.sin(2 * np.pi * 1 * t)

print(alias_frequency(9, fs))       # 1.0
print(np.allclose(x9, -x1))         # True
```

The nonnegative alias frequency is `1 Hz`. For these sine phases, the `9 Hz` samples match `-sin(2*pi*1*t)`; alias frequency describes the rate, while phase/sign must still be checked.

### Solution 6

```python
import numpy as np
import matplotlib.pyplot as plt

fs = 10.0
t_samples = np.arange(10) / fs
t_dense = np.linspace(0.0, 1.0, 3001)

x3_samples = np.sin(2 * np.pi * 3 * t_samples)
x13_samples = np.sin(2 * np.pi * 13 * t_samples)
print(np.allclose(x3_samples, x13_samples))  # True

plt.plot(t_dense, np.sin(2*np.pi*3*t_dense), label="3 Hz")
plt.plot(t_dense, np.sin(2*np.pi*13*t_dense), label="13 Hz", alpha=0.7)
plt.stem(t_samples, x3_samples, linefmt="k-", markerfmt="ko", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Because `13 = 3 + fs`, the two sample phases differ by `2*pi*n`. This is the lecture's ambiguity made visible.

### Solution 7

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, sinc_reconstruct


def signal(t):
    return np.sin(2 * np.pi * 3 * t)


fs = 10.0
t_samples, x_samples = sample_signal(signal, fs, -1.0, 1.1)
t_dense = np.linspace(-1.0, 1.0, 2001)
x_reconstructed = sinc_reconstruct(t_samples, x_samples, t_dense)
x_at_samples = sinc_reconstruct(t_samples, x_samples, t_samples)
print(np.allclose(x_at_samples, x_samples, atol=1e-12))  # True

plt.plot(t_dense, signal(t_dense), label="original")
plt.plot(t_dense, x_reconstructed, "--", label="finite sinc")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

The supplied samples are interpolated exactly. Any visible edge error comes from truncating the theoretical infinite sinc sum to a finite record.

### Solution 8

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, zero_order_hold


def signal(t):
    return np.cos(2 * np.pi * 2 * t)


fs = 10.0
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0, endpoint=True)
t_dense = np.linspace(0.0, 1.0, 2001)
x_zoh = zero_order_hold(t_samples, x_samples, t_dense)

plt.plot(t_dense, signal(t_dense), label="original")
plt.step(t_dense, x_zoh, where="post", label="ZOH")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Each stored value is held until the next sample. The staircase is causal and practical but is not ideal reconstruction.

### Solution 9

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import zoh_frequency_response

T = 0.1
f = np.linspace(-20.0, 20.0, 4001)
H0 = zoh_frequency_response(f, T)

dc_gain = abs(zoh_frequency_response([0.0], T)[0])
first_null = 1 / T
print(dc_gain)    # 0.1
print(first_null) # 10.0 Hz

plt.plot(f, np.abs(H0))
plt.axvline(first_null, color="C1", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H0(f)|")
plt.grid(alpha=0.3)
plt.show()
```

The ZOH has DC gain `0.1` and its first positive magnitude null at `10 Hz`. Between DC and the null, the sinc envelope causes passband droop.

### Solution 10

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import (
    magnitude_spectrum,
    meets_nyquist,
    sample_signal,
    sinc_reconstruct,
    zero_order_hold,
    zero_pad,
)


def signal(t):
    return np.sin(2*np.pi*4*t) + 0.4*np.cos(2*np.pi*11*t)


fs = 30.0
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0)
print(meets_nyquist(fs, 11.0))  # True: 30 > 22

positive_times = t_samples[x_samples > 0]
positive_values = x_samples[x_samples > 0]
print("positive sample count:", positive_values.size)

x_padded = zero_pad(x_samples, right=4)
print("padded length:", x_padded.size)  # 34

f, amplitude = magnitude_spectrum(x_samples, fs)
t_dense = np.linspace(t_samples[0], t_samples[-1], 2001)
x_sinc = sinc_reconstruct(t_samples, x_samples, t_dense)
x_zoh = zero_order_hold(t_samples, x_samples, t_dense)

fig, axes = plt.subplots(2, 1, figsize=(9, 7))
axes[0].plot(t_dense, signal(t_dense), label="original")
axes[0].plot(t_dense, x_sinc, "--", label="finite sinc")
axes[0].step(t_dense, x_zoh, where="post", label="ZOH")
axes[0].scatter(positive_times, positive_values, color="C3", label="positive samples")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].stem(f, amplitude)
axes[1].set(xlim=(0, fs/2), xlabel="Frequency (Hz)", ylabel="Amplitude")
axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()
```

Both `4 Hz` and `11 Hz` lie below `fs/2 = 15 Hz`, so the strict sampling condition holds. Sinc aims at ideal band-limited interpolation; ZOH produces the expected staircase and high-frequency droop.
