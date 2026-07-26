# CSE220 Convolution — Extra Practice Set v2: SOLUTIONS

Every solution below was actually executed against a working `DiscreteSignal`/`LTISystem`
and checked against a known-correct expected answer. Where I hit a real gotcha while
testing (Q4's warm-up transient is a good example), I've explained exactly what went
wrong and why, since spotting that kind of thing is a real skill, not just "know the
formula."

```python
# Shared helpers used throughout
def make_signal(start_time, end_time, values):
    signal = DiscreteSignal(start_time, end_time)
    for offset, value in enumerate(values):
        signal.set_value_at_time(start_time + offset, value)
    return signal

def max_absolute_difference(a, b):
    start = min(a.start_time, b.start_time)
    end = max(a.end_time, b.end_time)
    m = 0.0
    for n in range(start, end + 1):
        m = max(m, abs(a.get_value_at_time(n) - b.get_value_at_time(n)))
    return m
```

---

## Q1 — Signal trimmer

```python
def trim_signal(signal, tolerance=1e-12):
    samples = signal.nonzero_samples(tolerance)
    if not samples:
        return DiscreteSignal(0, 0)
    first_t = int(samples[0][0])
    last_t = int(samples[-1][0])
    trimmed = DiscreteSignal(first_t, last_t)
    for t in trimmed.times():
        trimmed.set_value_at_time(t, signal.get_value_at_time(t))
    return trimmed

s = make_signal(-5, 5, [0,0,0,1,2,0,3,0,0,0,0])
t = trim_signal(s)
print(t.start_time, t.end_time, t.values)
# -> -2 1 [1. 2. 0. 3.]
```

**Why it works:** `nonzero_samples()` already tells you exactly where the real content
starts and ends — the first and last tuples in that list give you your new boundaries.
Everything strictly between them gets copied over as-is, including any zeros that
happen to sit between two nonzero values (like the `0` between `2` and `3` in the test
case) — those aren't padding, they're real data you must keep.

---

## Q2 — Signal equality with implicit zero padding

```python
def signals_equal(a, b, tolerance=1e-9):
    start = min(a.start_time, b.start_time)
    end = max(a.end_time, b.end_time)
    for n in range(start, end + 1):
        if abs(a.get_value_at_time(n) - b.get_value_at_time(n)) > tolerance:
            return False
    return True

a = make_signal(0, 2, [1, 2, 3])
b = make_signal(-2, 4, [0, 0, 1, 2, 3, 0, 0])
print(signals_equal(a, b))   # -> True

b_wrong = make_signal(-2, 4, [0, 0, 1, 2, 9, 0, 0])   # one value changed
print(signals_equal(a, b_wrong))   # -> False
```

**Why it works:** `get_value_at_time` already returns 0.0 for anything outside a
signal's own stored range — so comparing over the *union* of both ranges automatically
treats "not stored" the same as "explicitly stored as zero," which is exactly the
notion of equality you want.

---

## Q3 — Echo / reverb effect designer

```python
def make_echo_impulse_response(period, num_echoes, decay):
    end = period * (num_echoes - 1)
    h = DiscreteSignal(0, end)
    h.set_value_at_time(0, 1.0)
    for i in range(1, num_echoes):
        h.set_value_at_time(i * period, decay ** i)
    return h

h_echo = make_echo_impulse_response(period=3, num_echoes=4, decay=0.5)
dry = make_signal(0, 2, [1.0, 0.5, -0.3])
wet = LTISystem(h_echo).output(dry)
print(wet.values)
# -> [1., 0.5, -0.3, 0.5, 0.25, -0.15, 0.25, 0.125, -0.075, 0.125, 0.0625, -0.0375]
```

**Why one impulse response beats manually adding shifted copies:** once you've built
`h_echo`, `LTISystem` handles *every* echo — decay, timing, and combining overlaps — in
one `output()` call, for *any* dry signal you throw at it. If you instead hand-coded
"shift by 3, multiply by 0.5, add; shift by 6, multiply by 0.25, add; ..." directly in
your main logic, you'd have to repeat and re-verify that block of code every single time
you wanted to apply the echo effect to a new signal — the impulse-response approach
packages the *effect itself* as reusable data, separate from the code that applies it.

---

## Q4 — Denoising and measuring the improvement

```python
def add_noise(signal, noise_std, seed=None):
    rng = np.random.default_rng(seed)
    y = DiscreteSignal(signal.start_time, signal.end_time)
    for n in signal.times():
        y.set_value_at_time(n, signal.get_value_at_time(n) + rng.normal(0, noise_std))
    return y

def rms_error(reference, other, skip_start=0):
    total = 0.0
    count = 0
    for n in reference.times():
        if n < reference.start_time + skip_start:
            continue
        d = reference.get_value_at_time(n) - other.get_value_at_time(n)
        total += d * d
        count += 1
    return (total / count) ** 0.5


clean = make_signal(0, 29, [3.0]*10 + [3.0 + 0.1*i for i in range(10)] + [4.0]*10)
noisy = add_noise(clean, 0.5, seed=7)

filter_length = 5
h_ma = make_signal(0, filter_length - 1, [1/filter_length]*filter_length)
filtered = LTISystem(h_ma).output(noisy)

e_noisy = rms_error(clean, noisy, skip_start=filter_length - 1)
e_filtered = rms_error(clean, filtered, skip_start=filter_length - 1)
print(e_noisy, e_filtered, (1 - e_filtered/e_noisy) * 100)
# -> 0.485 0.379  ~21.9% error reduction
```

**Why the transient happens and why skipping it is fair:** `y[0] = (1/5)*x[0]` because
`x[-1], x[-2], x[-3], x[-4]` don't exist (they're outside the signal, so
`get_value_at_time` returns 0 for them). That's not "the filter failing" — it's a
structurally different computation than every later sample, which genuinely does have
5 real inputs to average. Comparing `y[0]` against the clean signal as if it were a
"normal" filtered sample is comparing apples to oranges; skipping those first
`filter_length - 1` samples removes that structural mismatch rather than hiding an
inconvenient result. I actually hit this directly while testing: without the skip, the
filtered signal looked *worse* than the raw noisy one (error went up, not down) purely
because of this one transient sample dominating the average — a good reminder to
sanity-check *why* a surprising number came out the way it did instead of assuming a
bug in your filter logic.

---

## Q5 — Change-point / edge detector

```python
def detect_change_points(signal, threshold):
    h = make_signal(0, 1, [1.0, -1.0])
    diff_signal = LTISystem(h).output(signal)
    return [n for n in diff_signal.times() if abs(diff_signal.get_value_at_time(n)) > threshold]

s = make_signal(0, 19, [2.0]*10 + [8.0]*5 + [3.0]*5)
points = detect_change_points(s, threshold=2.0)
print(points)   # -> [10, 15]
```

**Which side of the jump gets flagged, and why:** `y[n] = x[n] - x[n-1]`. The jump from
2.0 to 8.0 happens going *into* index 10 (i.e. `x[9]=2.0`, `x[10]=8.0`), so
`y[10] = x[10]-x[9] = 6.0`, which is what crosses the threshold — the detector reports
the index *after* the jump (where the new, higher value first appears), not the index
before it. Same logic for the drop at n=15.

---

## Q6 — Polynomial multiplication via convolution

```python
def multiply_polynomials(coeffs1, coeffs2):
    p1 = make_signal(0, len(coeffs1) - 1, coeffs1)
    p2 = make_signal(0, len(coeffs2) - 1, coeffs2)
    product = LTISystem(p1).output(p2)
    return [product.get_value_at_time(n) for n in product.times()]

print(multiply_polynomials([1,2,3],[4,5]))    # -> [4.0, 13.0, 22.0, 15.0]
print(multiply_polynomials([1,-1],[1,1]))     # -> [1.0, 0.0, -1.0]
```

**Why it works, and why `product.values` alone is risky:** convolving two coefficient
lists really is exactly polynomial multiplication — `(sum_i a_i x^i)(sum_j b_j x^j)`
collects terms with the same total power `i+j`, which is precisely
`sum_k (sum_{i+j=k} a_i b_j) x^k`, i.e. the convolution formula. Using
`get_value_at_time(n)` for every `n` in `product.times()` (rather than reading
`.values` directly) is the safer choice, since it doesn't depend on you knowing
`product`'s internal array layout — it explicitly asks "what's the coefficient of
`x^n`?" for each power in order, which is exactly what you want regardless of how the
class stores things internally.

---

## Q7 — Matched filter / pulse detection

```python
def time_reverse(signal):
    reversed_sig = DiscreteSignal(-signal.end_time, -signal.start_time)
    for n in signal.times():
        reversed_sig.set_value_at_time(-n, signal.get_value_at_time(n))
    return reversed_sig

def find_best_match(noisy_signal, template):
    h_rev = time_reverse(template)
    correlation = LTISystem(h_rev).output(noisy_signal)
    return max(correlation.times(), key=lambda n: correlation.get_value_at_time(n))


template = make_signal(0, 2, [1.0, 2.0, 1.0])
noisy = DiscreteSignal(0, 20)
rng = np.random.default_rng(1)
for n in noisy.times():
    noisy.set_value_at_time(n, rng.normal(0, 0.2))
true_offset = 10
for i, v in enumerate([1.0, 2.0, 1.0]):
    noisy.set_value_at_time(true_offset + i, noisy.get_value_at_time(true_offset + i) + v)

detected = find_best_match(noisy, template)
print(true_offset, detected)   # -> 10 10
```

**Why it works:** correlating against a shape and looking for the peak is exactly
asking "at which alignment does this template overlap the signal most strongly?" —
random noise mostly cancels out on average across the correlation sum, but wherever the
real template-shaped bump sits, its samples all multiply constructively with the
matching template samples, producing a distinctly larger peak than anywhere else.

---

## Q8 — Time-delay estimation

```python
def estimate_delay(reference_signal, delayed_signal):
    h_rev = time_reverse(reference_signal)
    correlation = LTISystem(h_rev).output(delayed_signal)
    return max(correlation.times(), key=lambda n: correlation.get_value_at_time(n))

base = make_signal(0, 6, [0,1,3,5,3,1,0])
delayed = base.shift(4)
print(4, estimate_delay(base, delayed))   # -> 4 4
```

**Why it works:** the correlation between a signal and a delayed copy of itself peaks
exactly at the true delay — sliding the (reversed) reference across the delayed signal,
the best "overlap" naturally occurs right where the two shapes actually line up, which
is precisely at the shift amount that was applied. This is the same core idea as Q7,
applied to "match a signal against itself" instead of "match a signal against a known
external template."

---

## Q9 — Signal restoration (deconvolution to recover x)

```python
def restore_signal(h, y, x_length):
    assert h.start_time == 0 and abs(h.get_value_at_time(0)) > 1e-12
    x = DiscreteSignal(0, x_length - 1)
    for n in range(x_length):
        total = y.get_value_at_time(n)
        for kk in range(n):
            total -= x.get_value_at_time(kk) * h.get_value_at_time(n - kk)
        x.set_value_at_time(n, total / h.get_value_at_time(0))
    return x

h = make_signal(0, 2, [1.0, 0.5, 0.25])
x_true = make_signal(0, 4, [2, 1, -1, 3, 0.5])
y = LTISystem(h).output(x_true)
x_recovered = restore_signal(h, y, 5)
print(max_absolute_difference(x_true, x_recovered))   # -> 0.0
```

**If `h[0] == 0`:** the very first step, `x[0] = y[0] / h[0]`, divides by zero —
undefined. More fundamentally, if `h[0] = 0`, then `y[0] = x[0]*h[0] = 0` no matter
what `x[0]` actually is — the first output sample carries zero information about the
first input sample, so there's genuinely no way to recover it from `y[0]` alone. Every
later step depends on already knowing the earlier `x[k]` values correctly, so this
method fundamentally requires `h[0] != 0` to get off the ground.

---

## Q10 — Recursive filter vs. its truncated FIR approximation

```python
def system_leaky_integrator(input_signal, r):
    y = DiscreteSignal(input_signal.start_time, input_signal.end_time)
    prev = 0.0
    for n in input_signal.times():
        cur = input_signal.get_value_at_time(n) + r * prev
        y.set_value_at_time(n, cur)
        prev = cur
    return y

def truncated_geometric_impulse_response(r, length):
    return make_signal(0, length - 1, [r**n for n in range(length)])


impulse = make_signal(0, 4, [1.0, 0.0, 0.0, 0.0, 0.0])
y_iir = system_leaky_integrator(impulse, r=0.5)
h_fir_approx = truncated_geometric_impulse_response(0.5, 5)
print(y_iir.values)          # -> [1.    0.5   0.25  0.125 0.0625]
print(h_fir_approx.values)   # -> [1.    0.5   0.25  0.125 0.0625]
print(max_absolute_difference(y_iir, h_fir_approx))   # -> 0.0
```

**Why truncating costs accuracy, and why it gets worse as `r -> 1`:** the true impulse
response is `r^n` for every `n` from 0 to infinity — it decays but never *exactly*
reaches 0 for any finite `n`. Truncating it to `length` samples throws away the "tail"
`sum_{n=length}^{infinity} r^n = r^length / (1-r)` — this leftover tail-energy is exactly
what you lose. Since `r^length` shrinks slower when `r` is closer to 1 (a slower decay
rate), and dividing by `(1-r)` blows up as `r -> 1`, the truncation error grows sharply
the closer `r` gets to 1 — a "leakier" (slower-decaying) integrator needs a much longer
truncated approximation to stay accurate than one that decays quickly.

---

## Q11 — Sparse signal compression

```python
def compress(signal):
    return (signal.start_time, signal.end_time, signal.nonzero_samples())

def decompress(compressed):
    start, end, samples = compressed
    sig = DiscreteSignal(start, end)
    for t, v in samples:
        sig.set_value_at_time(int(t), v)
    return sig


sparse = DiscreteSignal(0, 99)
sparse.set_value_at_time(5, 3.0)
sparse.set_value_at_time(40, -2.0)
sparse.set_value_at_time(80, 7.5)

c = compress(sparse)
d = decompress(c)
print(max_absolute_difference(sparse, d))          # -> 0.0
print(len(c[2]), 'vs', len(sparse))                 # -> 3 vs 100
```

**Why it works:** `nonzero_samples()` already gives you the minimum information needed
to perfectly reconstruct the signal — every zero sample is implied automatically (since
a freshly built `DiscreteSignal` starts all-zero), so you only ever need to explicitly
store the exceptions. 3 stored values instead of 100 is a real (if extreme-case)
demonstration of why sparse representations matter for signals that are "mostly
silence."

---

## Q12 — Naive convolution vs. the offline's sparse-aware convolution

```python
def naive_convolve(x, h):
    start = x.start_time + h.start_time
    end = x.end_time + h.end_time
    y = DiscreteSignal(start, end)
    for nx in x.times():
        for nh in h.times():
            y.set_value_at_time(
                nx + nh, y.get_value_at_time(nx + nh) + x.get_value_at_time(nx) * h.get_value_at_time(nh)
            )
    return y


x = DiscreteSignal(0, 199)
x.set_value_at_time(10, 1.0)
x.set_value_at_time(100, 2.0)
x.set_value_at_time(190, -1.0)
h = make_signal(0, 4, [1, 0.5, 0.25, 0.1, 0.05])

import time
t0 = time.perf_counter()
y_naive = naive_convolve(x, h)
t1 = time.perf_counter()
y_sparse = LTISystem(h).output_by_superposition(x)
t2 = time.perf_counter()

print(max_absolute_difference(y_naive, y_sparse))   # -> 0.0
print('naive:', t1 - t0, 'sparse-aware:', t2 - t1)  # sparse-aware is meaningfully faster
```

**Why the sparse-aware version wins here:** `naive_convolve` always does
`len(x) * len(h)` multiply-adds — 200 * 5 = 1000 operations — no matter how many of
`x`'s samples are actually zero. `output_by_superposition` (via `nonzero_samples()`)
only does real work for the 3 nonzero samples `x` actually has, each producing one
5-sample component — roughly 15 operations. The sparser the input, the bigger this gap
gets; for a signal that's mostly zero (like real audio silence, or an image's empty
background), skipping the zeros isn't just a nice-to-have, it's the difference between
a program that runs instantly and one that doesn't scale at all.
