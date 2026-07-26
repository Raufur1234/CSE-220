# CSE220 Convolution — Extra Practice Set (13 Questions)

Built in the same style as the real CSE220 "Online on Convolution" (A1/A2): you reuse
your own `DiscreteSignal` and `LTISystem` classes from the offline assignment, write a
**generic** tester function that works for *any* `apply_system` callable (not just
`LTISystem`), then apply it to concrete systems. `max_absolute_difference` is assumed
available exactly as in the real exam.

**Rules (same as the real thing):**
- Do NOT use `numpy.convolve`, `scipy.signal`, or any built-in convolution.
- Every "tester" function must work for any `apply_system(signal) -> signal` callable —
  do not assume it's an `LTISystem`.
- Put your file next to `signal_lti.py` so `from signal_lti import DiscreteSignal, LTISystem`
  works.

Difficulty increases as you go. Questions 1–3 are exam-length (~10–15 min each).
Questions 10–13 are genuinely hard — don't expect to finish them in 15 minutes on
your first attempt.

---

## Shared setup (reuse across all questions unless a question overrides it)

```python
x1 = make_signal(-2, 2, [1, 0, 2, -1, 3])
x2 = make_signal(-1, 3, [2, -3, 0, 1, 1])
a, b = 2.0, -3.0
k = 3

h  = make_signal(0, 2, [1.0, 0.5, 0.25])     # a causal FIR impulse response
h1 = make_signal(0, 1, [1.0, -0.5])          # a second, shorter impulse response
h2 = make_signal(0, 2, [0.5, 0.25, 0.1])     # a third impulse response
```

---

## Q1 — Causality tester (10 marks, ~12 min)

A system is **causal** if the output at time `n0` depends only on input values at times
`n <= n0` — nothing from the future leaks in.

**Task:** Implement a generic tester:

```python
def test_causality(apply_system, x, n0):
    """
    Build a second input signal that is IDENTICAL to x for all n <= n0, but
    different from x for n > n0 (e.g. add some constant to every future sample).
    Run apply_system on both. If the system is causal, the two outputs must
    match for every n <= n0, no matter what you changed in the future.
    Return the maximum absolute difference between the two outputs, but ONLY
    checked over n <= n0 (this is important — differences after n0 are expected
    and should not be counted).
    """
    # TODO
```

Test it on:
1. `system_causal = LTISystem(h).output` (should be causal — h starts at n=0)
2. `system_lookahead(x)`, defined as `y[n] = x[n+1]` (a "peek 1 sample into the future"
   system — you must implement this one yourself too)

Print both max-diff values at `n0 = 0`, and state in a print statement which system is
causal and which is not, and why.

**Hint:** you don't need `n0` to be special — pick it, then only compare `y1` and `y2`
over the range `x.start_time` to `n0` inclusive.

---

## Q2 — Memorylessness tester (10 marks, ~12 min)

A system is **memoryless** (has "no memory") if the output at time `n0` depends *only*
on the input value at that exact same time `n0` — not on any neighboring samples.

**Task:** Implement:

```python
def test_memoryless(apply_system, x, n0):
    """
    Build a second signal identical to x everywhere EXCEPT keep x[n0] the same
    and change every OTHER sample (add some constant to them). Run apply_system
    on both. If the system is memoryless, y[n0] must be unchanged, since only
    x[n0] itself matters for that output sample.
    Return |y1[n0] - y2[n0]|.
    """
    # TODO
```

Test it on:
1. `system_square(x)`: `y[n] = x[n]^2` (should be memoryless — implement this system too)
2. `LTISystem(h_ma).output` where `h_ma` is a 3-point moving average (`[1/3, 1/3, 1/3]`)
   (should NOT be memoryless — any real filter with more than one nonzero tap has memory)

Print both results and state which system has memory.

---

## Q3 — BIBO stability bound (10 marks, ~15 min)

A system is **BIBO stable** ("bounded input, bounded output") if every bounded input
produces a bounded output. For an LTI system with impulse response `h[n]`, there's a
famous fact: the system is BIBO stable if and only if `sum(|h[n]|)` is finite, and that
sum is also the *tightest possible gain bound* — i.e. `max|y[n]| <= sum(|h[n]|) * max|x[n]|`
always holds.

**Task:**
```python
def bibo_gain_bound(h):
    """Return sum(|h[n]|) over all nonzero samples of h."""
    # TODO

def empirical_worst_case_gain(system, num_trials, min_len, max_len, amplitude):
    """
    Run num_trials random experiments. Each trial: pick a random signal length
    between min_len and max_len, fill it with random values in
    [-amplitude, amplitude], run it through `system`, and compute
    max|y[n]| / max|x[n]|. Return the largest ratio you ever observed.
    """
    # TODO
```

Use `h` from the shared setup. Print the theoretical `bibo_gain_bound(h)` and the
`empirical_worst_case_gain` over at least 200 random trials. State in a print
statement whether the empirical value respects the theoretical bound (it always
should — if it doesn't, you have a bug).

---

## Q4 — Convolution is commutative: x * h = h * x (10 marks, ~10 min)

Convolution doesn't care which signal you call "the input" and which you call "the
impulse response" — the math is symmetric.

**Task:** Using only `LTISystem`, show that convolving `x` with `h` gives the same
answer as convolving `h` with `x`:

```python
def test_commutativity(x, h):
    """
    Build LTISystem(h) and apply it to x -- that's x * h.
    Build LTISystem(x) and apply it to h -- that's h * x.
    Return the maximum absolute difference between the two results.
    """
    # TODO
```

Run it on `x1` and `h`. Print the result — it should be extremely close to 0
(floating-point noise at worst, e.g. `1e-16`).

---

## Q5 — Convolution is associative: (x * h1) * h2 = x * (h1 * h2) (15 marks, ~15 min)

If you send a signal through one filter, then through a second filter (a **cascade**),
that's mathematically identical to combining the two filters into one single impulse
response first (by convolving them together), and applying that combined filter once.

**Task:**
```python
def test_associativity(x, h1, h2):
    """
    Path 1 (cascade): apply LTISystem(h1) to x, then apply LTISystem(h2) to
    that result.
    Path 2 (pre-combine): compute h_combined = h1 convolved with h2 (hint: you
    can literally reuse LTISystem to convolve two signals together -- an
    LTISystem's impulse response is just a DiscreteSignal, and so is h2).
    Then build LTISystem(h_combined) and apply it to x once.
    Return the maximum absolute difference between the two paths' outputs.
    """
    # TODO
```

Run on `x1`, `h1`, `h2` from the shared setup. Print the result and a one-line
explanation of why this proves cascading two LTI filters is equivalent to a single
combined filter.

---

## Q6 — Convolution distributes over addition: x*(h1+h2) = x*h1 + x*h2 (15 marks, ~15 min)

This is the mathematical fact behind **parallel** filter combination: running a signal
through two filters separately and adding the results is the same as adding the two
filters together first and running the signal through the combined filter once.

**Task:**
```python
def test_distributivity(x, h1, h2):
    """
    Path 1 (parallel): apply LTISystem(h1) to x, apply LTISystem(h2) to x
    separately, add the two results together.
    Path 2 (pre-sum): compute h_sum = h1.add(h2), build LTISystem(h_sum),
    apply it to x once.
    Return the maximum absolute difference between the two paths.
    """
    # TODO
```

Run on `x1`, `h1`, `h2`. Print the result and state, in one sentence, a real-world
reason you might prefer combining filters into one (Path 2) instead of running them
in parallel and adding (Path 1).

---

## Q7 — Recovering h[n] from the step response (15 marks, ~15 min)

The **unit step** `u[n]` is 1 for `n >= 0` and 0 for `n < 0`. The **step response**
`s[n]` of an LTI system is what you get when you feed it `u[n]`. There's a classic
relationship: `s[n]` is the *running sum* of `h[n]`, and conversely
`h[n] = s[n] - s[n-1]` (the first difference of the step response recovers the
impulse response exactly).

**Task:**
```python
def make_unit_step(start_time, end_time):
    """Build a DiscreteSignal that is 0 for n < 0 and 1 for n >= 0, over the
    given range. (You need start_time to be safely negative and end_time
    safely larger than h's support, so the step response has "settled" and
    you're not cutting off any of h's contribution.)"""
    # TODO

def recover_impulse_response(step_response):
    """Given a DiscreteSignal step response, compute h[n] = s[n] - s[n-1]
    for every n in its range, and return it as a new DiscreteSignal."""
    # TODO
```

Using `h` from the shared setup: build `LTISystem(h)`, generate its step response
using a unit step spanning at least `[-3, 8]`, recover `h` from the step response,
and print `max_absolute_difference` between the recovered `h` and the true `h`
(only compare over the range where `h` is actually defined, `n = 0..2`).

**Warning (this trips people up):** if your unit step window is too narrow or starts
at 0 instead of safely negative, your recovered `h` will be wrong at the edges. Think
about why before you debug blindly.

---

## Q8 — Signal energy and a "gain-preserves-relative-energy" check (10 marks, ~10 min)

The **energy** of a discrete signal is `sum(x[n]^2)` over all its samples (only the
nonzero ones matter, since squaring zero is still zero).

**Task:**
```python
def signal_energy(signal):
    """Return sum(value^2) over every stored sample of the signal."""
    # TODO

def test_energy_scaling(x, scalar):
    """
    A basic sanity fact: scaling every sample of x by `scalar` should scale
    the ENERGY by scalar^2 (since energy involves squaring).
    Compute signal_energy(x.multiply(scalar)) and compare it against
    signal_energy(x) * (scalar ** 2). Return the absolute difference.
    """
    # TODO
```

Run `test_energy_scaling(x1, 2.5)` and print the result — it should be extremely
close to 0.

---

## Q9 — Cross-correlation, computed using ONLY convolution primitives (20 marks, ~20 min)

Cross-correlation `r[lag] = sum_n x[n] * h[n - lag]` measures how well `h` matches `x`
at different offsets — the basis of matched filtering, pattern detection, etc. There's
a neat trick: correlation is just convolution with one signal **time-reversed** first.

**Task:**
```python
def time_reverse(signal):
    """Return a new DiscreteSignal y such that y[n] = signal[-n] for every n.
    (Hint: think about what happens to start_time and end_time when you flip
    every time label's sign -- the earliest time becomes the latest, negated.)"""
    # TODO

def correlate_via_convolution(x, h):
    """Compute the cross-correlation of x and h using ONLY DiscreteSignal /
    LTISystem operations: time-reverse h, then convolve x with the
    reversed signal using LTISystem."""
    # TODO

def correlate_directly(x, h):
    """A brute-force direct implementation for comparison: for every possible
    lag in the appropriate range, compute sum_n x[n] * h[n - lag] directly
    with a nested loop (no shortcuts). Return it as a DiscreteSignal."""
    # TODO
```

Run both on `x1` and `h1`, print `max_absolute_difference` between them (should be
~0), confirming the "correlation = convolution with a flipped signal" trick works.

---

## Q10 — FIR system identification (deconvolution) (25 marks, ~25 min, hard)

Suppose you know the input `x[n]` (causal, starting at `n=0`, with `x[0] != 0`) and
the output `y[n] = (x * h)[n]` of some unknown FIR system, and you know `h` has exactly
`h_length` samples. You can recover `h` sample by sample, using the fact that
`y[n] = sum_{k=0}^{n} x[n-k] h[k]`, which rearranges to:

```
h[0] = y[0] / x[0]
h[n] = ( y[n] - sum_{k=0}^{n-1} x[n-k] * h[k] ) / x[0]      for n = 1, 2, ...
```

**Task:**
```python
def deconvolve(x, y, h_length):
    """
    x: a causal DiscreteSignal with x.start_time == 0 and x[0] != 0.
    y: the known output signal (x convolved with the unknown h).
    h_length: how many samples the unknown h has (starting at n=0).
    Return the recovered h as a DiscreteSignal over [0, h_length - 1],
    using the formula above.
    """
    # TODO
```

**Test it properly (this is the important part):** pick some `x` (causal, `x[0] != 0`),
invent a "true" `h` of your choice, use `LTISystem(h_true).output(x)` to generate `y`
(this is your ground truth), then call `deconvolve(x, y, len(h_true))` and confirm
`max_absolute_difference` between the recovered `h` and `h_true` is ~0.

Explain in a comment why this method requires `x[0] != 0` — what would go wrong
otherwise?

---

## Q11 — A recursive (IIR) system: is it linear? Is it time-invariant? (25 marks, ~25 min, hard)

All the impulse responses so far have been finite (FIR). Real systems are often
**recursive**: the output depends on *previous outputs*, not just inputs. Consider:

```
y[n] = x[n] + 0.5 * y[n-1]      (with "initial rest": y[n] = 0 for all n before
                                  the input signal's start_time)
```

**Task:**
```python
def system_iir(input_signal):
    """
    Implement y[n] = x[n] + 0.5*y[n-1] using ONLY a single forward loop over
    input_signal.times() and a running "previous output" variable that starts
    at 0.0 (representing initial rest). Return the result as a DiscreteSignal
    over the same range as input_signal.
    """
    # TODO
```

Using your `test_linearity` and `test_time_invariance` testers from earlier
(rewrite them here if you're doing this standalone), test `system_iir` using
`x1`, `x2`, `a`, `b`, `k` from the shared setup.

**Important subtlety to think about and explain in a comment:** for these results to
come out as (essentially) zero, `x1` and `x2` need to have their `start_time` set to
the *same* value before you test linearity. Why? (Hint: think about what "initial
rest" means, and where each signal's recursion actually "starts" relative to its own
`start_time`.) Try it with `x1` and `x2` having different `start_time`s (as given in
the shared setup) and see what happens to the linearity result — then fix it by
rebuilding `x2` over the same range as `x1` and explain the difference you observe.

---

## Q12 — A nonlinear, memoryless, time-invariant system (10 marks, ~10 min)

Not every "broken" system breaks in the same way. Squaring each sample,
`y[n] = x[n]^2`, is a common example of a system that is **not linear**, but IS
time-invariant (delaying the input just delays the squared output by the same amount)
and IS memoryless (no neighboring samples involved).

**Task:** Implement `system_square(x)` (`y[n] = x[n]^2`) and run both
`test_linearity` and `test_time_invariance` on it using `x1`, `x2`, `a`, `b`, `k`.
Print both results and, in a print statement, state clearly which property (if any)
this system fails, and confirm your intuition matches the numeric result.

---

## Q13 — A linear-but-time-varying system, and a subtlety about finite signals (20 marks, ~20 min, hardest / discussion)

Consider `y[n] = x[n] + x[-n]` (add each sample to the sample that's the same distance
on the *other side* of time zero). Mathematically, for signals with infinite support,
this system is **linear** (you can check: distributing `a` and `b` through it works out
algebraically) but **not time-invariant** (delaying the input changes which samples
get reflected around a fixed point n=0 — the reflection point doesn't move with the
signal).

**Task:**
```python
def system_reflect(x):
    """Implement y[n] = x[n] + x[-n] over the same range as x. Remember
    get_value_at_time returns 0.0 automatically for times outside x's range,
    so you don't need to handle "out of range" as a special case yourself."""
    # TODO
```

Run `test_linearity(system_reflect, x1, x2, a, b)` using the shared `x1`, `x2`, `a`, `b`.
You will most likely get a **nonzero** result, even though the underlying math (for a
true infinite-support signal) says this system should be linear.

**Your job:** figure out why, and explain it in 2-3 sentences in a comment. (Hint:
print out `x1`'s range, `x2`'s range, and the range of `a*x1 + b*x2` — are they all the
same? What happens when `system_reflect` is applied to a signal whose range is wider
than either `x1` or `x2` alone, versus applied to `x1` and `x2` separately and then
added?) This is a real, important lesson about the difference between the true
mathematical definition of a system and what happens when you're forced to represent
signals with finite storage.
