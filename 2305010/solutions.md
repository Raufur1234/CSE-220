# CSE220 Convolution — Extra Practice Set: SOLUTIONS

Every solution below was actually run against a working `DiscreteSignal`/`LTISystem`
implementation and verified to produce the expected numeric result (stated under each
question). Don't just copy these — trace through the logic by hand on a tiny example
first (like we did earlier for `x=[1,2]`, `h=[5,1]`), otherwise you'll be lost the moment
the real exam changes a number.

```python
# Shared setup used throughout
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

x1 = make_signal(-2, 2, [1, 0, 2, -1, 3])
x2 = make_signal(-1, 3, [2, -3, 0, 1, 1])
a, b = 2.0, -3.0
k = 3
h  = make_signal(0, 2, [1.0, 0.5, 0.25])
h1 = make_signal(0, 1, [1.0, -0.5])
h2 = make_signal(0, 2, [0.5, 0.25, 0.1])
```

---

## Q1 — Causality tester

```python
def test_causality(apply_system, x, n0):
    # Build a copy of x that agrees with x up to n0, but is different afterward
    x_mod = DiscreteSignal(x.start_time, x.end_time)
    for n in x.times():
        if n <= n0:
            x_mod.set_value_at_time(n, x.get_value_at_time(n))
        else:
            x_mod.set_value_at_time(n, x.get_value_at_time(n) + 100.0)

    y1 = apply_system(x)
    y2 = apply_system(x_mod)

    # Only compare the outputs up to n0 -- differences AFTER n0 don't matter
    max_diff = 0.0
    for n in range(x.start_time, n0 + 1):
        max_diff = max(max_diff, abs(y1.get_value_at_time(n) - y2.get_value_at_time(n)))
    return max_diff


def system_lookahead(sig):
    y = DiscreteSignal(sig.start_time, sig.end_time)
    for n in sig.times():
        y.set_value_at_time(n, sig.get_value_at_time(n + 1))
    return y


sys_causal = LTISystem(h)
print(test_causality(sys_causal.output, x1, 0))       # -> 0.0
print(test_causality(system_lookahead, x1, 0))         # -> 100.0
```

**Why it works:** `LTISystem(h).output` is causal because `h` only has nonzero samples
at `n >= 0` — every output `y[n]` only ever multiplies `x[k]` for `k <= n` (recall
`y[n] = sum_k x[k]*h[n-k]`, and `h[n-k]` is only nonzero when `n-k >= 0`, i.e. `k <= n`).
`system_lookahead` explicitly reads `x[n+1]` — one sample into the future — so changing
future samples of `x` (at `n = 1`) changes `y[0] = x[1]`, producing exactly the injected
`100.0` difference.

---

## Q2 — Memorylessness tester

```python
def test_memoryless(apply_system, x, n0):
    x_mod = DiscreteSignal(x.start_time, x.end_time)
    for n in x.times():
        if n == n0:
            x_mod.set_value_at_time(n, x.get_value_at_time(n))
        else:
            x_mod.set_value_at_time(n, x.get_value_at_time(n) + 50.0)

    y1 = apply_system(x)
    y2 = apply_system(x_mod)
    return abs(y1.get_value_at_time(n0) - y2.get_value_at_time(n0))


def system_square(sig):
    y = DiscreteSignal(sig.start_time, sig.end_time)
    for n in sig.times():
        v = sig.get_value_at_time(n)
        y.set_value_at_time(n, v * v)
    return y


print(test_memoryless(system_square, x1, 0))                 # -> 0.0
h_ma = make_signal(0, 2, [1/3, 1/3, 1/3])
print(test_memoryless(LTISystem(h_ma).output, x1, 0))         # -> 33.33...
```

**Why it works:** `system_square` only ever looks at `x[n]` to compute `y[n]` — changing
every *other* sample can't touch `y[n0]`, so the diff is exactly 0. The moving-average
filter blends 3 neighboring samples together, so `y[0]` depends on `x[-1], x[0], x[1]` —
changing `x[-1]` and `x[1]` by +50 each changes `y[0]` by `50/3 + 50/3 = 33.33`, matching
the observed diff.

---

## Q3 — BIBO stability bound

```python
def bibo_gain_bound(h_signal):
    return sum(abs(v) for _, v in h_signal.nonzero_samples())


def empirical_worst_case_gain(system, num_trials, min_len, max_len, amplitude):
    import random
    worst = 0.0
    for _ in range(num_trials):
        n = random.randint(min_len, max_len)
        vals = [random.uniform(-amplitude, amplitude) for _ in range(n)]
        xsig = make_signal(0, n - 1, vals)
        ysig = system(xsig)
        input_max = max((abs(v) for v in xsig.values), default=0.0)
        output_max = max((abs(v) for v in ysig.values), default=0.0)
        if input_max > 0:
            worst = max(worst, output_max / input_max)
    return worst


sys_h = LTISystem(h)
bound = bibo_gain_bound(h)                                          # -> 1.75
empirical = empirical_worst_case_gain(sys_h.output, 200, 3, 10, 5.0)
print(bound, empirical)             # e.g. 1.75  1.71   (empirical is ALWAYS <= bound)
```

**Why it works:** `|y[n]| = |sum_k x[k]h[n-k]| <= max|x| * sum_k|h[n-k]| <= max|x| * sum|h|`
— that's a direct algebraic bound, so no amount of random testing can ever exceed it
(you can get *close* with the right input shape, but never over).

---

## Q4 — Commutativity

```python
def test_commutativity(x, h_signal):
    out1 = LTISystem(h_signal).output(x)   # x * h
    out2 = LTISystem(x).output(h_signal)   # h * x
    return max_absolute_difference(out1, out2)

print(test_commutativity(x1, h))    # -> ~1e-16 (floating point noise, effectively 0)
```

**Why it works:** `y[n] = sum_k x[k]h[n-k]`. Substitute `j = n-k` (so `k = n-j`):
`y[n] = sum_j x[n-j]h[j]` — that's exactly the convolution formula with the roles of
`x` and `h` swapped. It's the same sum, just walked in the opposite order.

---

## Q5 — Associativity (cascade filters)

```python
def test_associativity(x, h_a, h_b):
    # Path 1: cascade -- x through h_a, then through h_b
    stage1 = LTISystem(h_a).output(x)
    cascade_out = LTISystem(h_b).output(stage1)

    # Path 2: combine h_a and h_b first (convolve them together), apply once
    h_combined = LTISystem(h_a).output(h_b)   # treats h_b AS the "input signal"
    direct_out = LTISystem(h_combined).output(x)

    return max_absolute_difference(cascade_out, direct_out)

print(test_associativity(x1, h1, h2))   # -> ~5.5e-17 (effectively 0)
```

**Why it works:** convolution is associative: `(x*h_a)*h_b = x*(h_a*h_b)`. The trick
in the code is realizing `LTISystem(h_a).output(h_b)` computes `h_a * h_b` — you don't
need a separate "convolve two arbitrary signals" function, `LTISystem` already IS that,
since its impulse response and its input are both just `DiscreteSignal` objects.

---

## Q6 — Distributivity (parallel filters)

```python
def test_distributivity(x, h_a, h_b):
    # Path 1: parallel -- run x through h_a and h_b separately, add results
    parallel_out = LTISystem(h_a).output(x).add(LTISystem(h_b).output(x))

    # Path 2: add the filters first, run x through the combined filter once
    h_sum = h_a.add(h_b)
    combined_out = LTISystem(h_sum).output(x)

    return max_absolute_difference(parallel_out, combined_out)

print(test_distributivity(x1, h1, h2))   # -> 0.0
```

**Why it works:** `x*(h_a+h_b) = x*h_a + x*h_b` is just the distributive law applied
inside the convolution sum. Practically: if you're running two filters in parallel and
adding their outputs every single time, you can save computation by pre-adding the
filters once and only running one convolution per input afterward.

---

## Q7 — Recovering h[n] from the step response

```python
def make_unit_step(start_time, end_time):
    u = DiscreteSignal(start_time, end_time)
    for n in u.times():
        u.set_value_at_time(n, 1.0 if n >= 0 else 0.0)
    return u


def recover_impulse_response(step_response):
    recovered = DiscreteSignal(step_response.start_time, step_response.end_time)
    for n in recovered.times():
        recovered.set_value_at_time(
            n, step_response.get_value_at_time(n) - step_response.get_value_at_time(n - 1)
        )
    return recovered


sys_h = LTISystem(h)
u = make_unit_step(-3, 8)
s = sys_h.output(u)
h_recovered = recover_impulse_response(s)
print(max_absolute_difference(h, h_recovered))   # -> 0.0 (checked over h's own range)
```

**Why the warning in the question matters:** if the step doesn't start safely negative
(e.g. it starts at exactly `n=0` and is "1 everywhere in its window" instead of properly
0-before-0), then `s[-1]` isn't really 0 like a true step response would give you — it's
undefined/wrong, which throws off `h[0] = s[0] - s[-1]`. Always give the step signal
padding before `n=0` so the "0 before the step" behavior is actually represented.

---

## Q8 — Signal energy

```python
def signal_energy(signal):
    return sum(v * v for _, v in signal.nonzero_samples())


def test_energy_scaling(x, scalar):
    scaled_energy = signal_energy(x.multiply(scalar))
    predicted_energy = signal_energy(x) * (scalar ** 2)
    return abs(scaled_energy - predicted_energy)

print(test_energy_scaling(x1, 2.5))   # -> 0.0 (or extremely close, floating point)
```

**Why it works:** `sum((c*x[n])^2) = c^2 * sum(x[n]^2)` — squaring distributes the
scalar out as `c^2`, pure algebra, nothing signal-specific about it.

---

## Q9 — Cross-correlation via convolution

```python
def time_reverse(signal):
    reversed_sig = DiscreteSignal(-signal.end_time, -signal.start_time)
    for n in signal.times():
        reversed_sig.set_value_at_time(-n, signal.get_value_at_time(n))
    return reversed_sig


def correlate_via_convolution(x, h_signal):
    h_rev = time_reverse(h_signal)
    return LTISystem(h_rev).output(x)


def correlate_directly(x, h_signal):
    start = x.start_time - h_signal.end_time
    end = x.end_time - h_signal.start_time
    result = DiscreteSignal(start, end)
    for lag in result.times():
        total = 0.0
        for n in x.times():
            total += x.get_value_at_time(n) * h_signal.get_value_at_time(n - lag)
        result.set_value_at_time(lag, total)
    return result


r1 = correlate_via_convolution(x1, h1)
r2 = correlate_directly(x1, h1)
print(max_absolute_difference(r1, r2))   # -> 0.0
```

**Why it works:** convolution is `y[n] = sum_k x[k]h[n-k]`. If you replace `h` with its
time-reversed version `h_rev[n] = h[-n]`, then `h_rev[n-k] = h[-(n-k)] = h[k-n]`, and
`sum_k x[k]h[k-n]` is exactly the correlation formula (with a sign flip on the lag
variable, which is why `time_reverse` needs to correctly swap and negate the start/end
times — get this wrong and your ranges won't line up even if the values are right).

---

## Q10 — FIR deconvolution (system identification)

```python
def deconvolve(x, y, h_length):
    assert x.start_time == 0 and abs(x.get_value_at_time(0)) > 1e-12
    h_recovered = DiscreteSignal(0, h_length - 1)
    for n in range(h_length):
        total = y.get_value_at_time(n)
        for kk in range(n):
            total -= x.get_value_at_time(n - kk) * h_recovered.get_value_at_time(kk)
        h_recovered.set_value_at_time(n, total / x.get_value_at_time(0))
    return h_recovered


x_test = make_signal(0, 4, [2, 1, -1, 3, 0.5])
h_true = make_signal(0, 2, [1.0, 0.5, 0.25])
y_test = LTISystem(h_true).output(x_test)

h_found = deconvolve(x_test, y_test, 3)
print(max_absolute_difference(h_true, h_found))   # -> 0.0
```

**Why `x[0] != 0` matters:** the very first step, `h[0] = y[0]/x[0]`, divides by `x[0]`.
If `x[0] == 0`, that's division by zero — undefined. More fundamentally, if `x[0] = 0`,
then `y[0] = x[0]*h[0] = 0` regardless of what `h[0]` actually is — there's no way to
recover `h[0]` from `y[0]` in that case, because the very first output sample carries no
information about it. Every later step in the recursion relies on knowing all previous
`h[k]` values correctly, so an unrecoverable `h[0]` poisons the entire result.

---

## Q11 — Recursive (IIR) system, linearity & time-invariance

```python
def system_iir(input_signal):
    y = DiscreteSignal(input_signal.start_time, input_signal.end_time)
    previous_output = 0.0   # initial rest: y = 0 before the signal starts
    for n in input_signal.times():
        current = input_signal.get_value_at_time(n) + 0.5 * previous_output
        y.set_value_at_time(n, current)
        previous_output = current
    return y


def test_linearity(apply_system, xa, xb, aa, bb):
    out1 = apply_system(xa.multiply(aa).add(xb.multiply(bb)))
    out2 = apply_system(xa).multiply(aa).add(apply_system(xb).multiply(bb))
    return max_absolute_difference(out1, out2)


def test_time_invariance(apply_system, xsig, kk):
    out1 = apply_system(xsig.shift(kk))
    out2 = apply_system(xsig).shift(kk)
    return max_absolute_difference(out1, out2)


# IMPORTANT: rebuild x2 to share x1's start_time before testing linearity
x2_aligned = make_signal(x1.start_time, x1.start_time + 4, [2, -3, 0, 1, 1])

print(test_linearity(system_iir, x1, x2_aligned, a, b))   # -> 0.0
print(test_time_invariance(system_iir, x1, k))             # -> 0.0
```

**Why the subtlety matters:** "initial rest" means `previous_output` starts at 0 exactly
at `input_signal.start_time` — that's the recursion's true "beginning." If `x1` starts
at `n=-2` and `x2` starts at `n=-1`, their recursions begin at *different absolute times*.
When you combine `a*x1 + b*x2`, the combined signal's recursion begins at the earliest of
the two starting points — but `apply_system(x1)` and `apply_system(x2)` each independently
"reset to 0" at their *own* different starting points. Those resets don't line up, so the
two computation paths silently diverge even though the math should agree. Rebuilding `x2`
to start at the same time as `x1` removes that mismatch and the test passes cleanly.
This is a genuinely important, often-overlooked fact about testing recursive systems:
alignment of "where the recursion starts" matters just as much as the sample values.

---

## Q12 — Nonlinear but time-invariant system

```python
def system_square(sig):
    y = DiscreteSignal(sig.start_time, sig.end_time)
    for n in sig.times():
        v = sig.get_value_at_time(n)
        y.set_value_at_time(n, v * v)
    return y

print(test_linearity(system_square, x1, x2, a, b))         # -> 188.0 (fails)
print(test_time_invariance(system_square, x1, k))           # -> 0.0   (passes)
```

**Why:** `(x[n])^2` is a nonlinear operation (squaring doesn't distribute over addition
or scaling: `(a*p + b*q)^2 != a*p^2 + b*q^2` in general) — so linearity fails hard (188.0
is a large diff, not floating-point noise). But shifting the input by `k` and then
squaring gives the exact same result as squaring first and then shifting — squaring
each sample doesn't care *when* that sample occurs, only *what its value is* — so
time-invariance holds exactly.

---

## Q13 — Linear but time-varying system, and the finite-signal subtlety

```python
def system_reflect(x):
    y = DiscreteSignal(x.start_time, x.end_time)
    for n in x.times():
        y.set_value_at_time(n, x.get_value_at_time(n) + x.get_value_at_time(-n))
    return y

print(test_linearity(system_reflect, x1, x2, a, b))   # -> 3.0 (NOT 0, surprisingly!)
```

**Why this happens (the actual lesson):** mathematically, for a signal with infinite
support, `y[n] = x[n] + x[-n]` genuinely is linear. But `x1` and `x2` here are *finite*
signals, each only defined over its own narrow range (`x1`: -2..2, `x2`: -1..3). When
`system_reflect` is applied to the **combined** signal `a*x1 + b*x2` (whose range is the
*union* of both ranges, -2..3), it can "see" and reflect values from that whole wider
combined window. But when `system_reflect` is applied to `x2` **by itself**, its output
is only defined over `x2`'s own narrow range (-1..3) — anything it would need to look up
outside that range (like reflecting into `n=-2`, which is inside `x1`'s range but outside
`x2`'s) silently returns 0 instead of the "correct" value, because `get_value_at_time`
treats "outside my own stored range" as zero, with no way to know that some *other*
signal happens to have real data there.

In short: **the two computation paths don't actually have access to the same
information window**, purely because of how finite-length signals are stored and
truncated — it's an artifact of finite representation, not a real violation of
linearity for the underlying infinite-signal mathematics. This is worth understanding
deeply: it's exactly the kind of edge case where "the math says X" and "the code says
not-quite-X" for a completely understandable, explainable reason — and being able to
explain *why*, rather than just reporting a number, is what separates a strong answer
from a mediocre one on a question like this.
