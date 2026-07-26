# CSE220 Convolution — Extra Practice Set v2 (Revamped, Varied Tasks)

This set drops the "every question is a linearity/time-invariance tester" pattern from
the real A1/A2 online. Instead, each question is a **different kind of task** — signal
processing, restoration, detection, even things unrelated to filtering at first glance
(like polynomial multiplication) — but every single one is solved using ONLY your
`DiscreteSignal` / `LTISystem` classes from the offline. The point is to get comfortable
reusing that machinery in unfamiliar contexts, since that's what a real exam question
can throw at you.

**Rules (same as always):**
- No `numpy.convolve`, `scipy.signal`, or any built-in convolution — build everything
  from `DiscreteSignal`/`LTISystem` operations (`shift`, `multiply`, `add`,
  `nonzero_samples`, `get_value_at_time`, `set_value_at_time`, `LTISystem.output`, etc.)
- Put your file next to `signal_lti.py`.

Roughly ordered easy → hard, but they test different skills, not just "harder version
of the same skill" — read each one fresh.

---

## Q1 — Signal trimmer (10 marks, ~10 min)

A signal built with `DiscreteSignal(start, end)` might have a lot of wasted zero-padding
at the edges (e.g. spans -5..5 but the only real data is at -2..1). Write a function
that shrinks a signal down to just the range it actually needs.

```python
def trim_signal(signal, tolerance=1e-12):
    """
    Return a NEW DiscreteSignal spanning only from the first nonzero sample to
    the last nonzero sample (inclusive), preserving all the values in between
    (including any zeros that fall between two nonzero samples -- those stay).
    If the signal is entirely zero, return a signal with a single sample
    at time 0 (edge case, handle it however seems sensible, but don't crash).
    Hint: nonzero_samples() already gives you exactly the boundary information
    you need -- you don't need to write your own scanning loop from scratch.
    """
    # TODO
```

Test it on a signal spanning -5..5 with values `[0,0,0,1,2,0,3,0,0,0,0]` (i.e. real
data only at times -2, -1, 0, 1). Print the trimmed signal's `start_time`, `end_time`,
and values, and confirm by hand that it matches what you'd expect.

---

## Q2 — Signal equality checker, ignoring zero-padding (10 marks, ~10 min)

Two `DiscreteSignal` objects can represent the *exact same signal* even if they were
built with completely different `start_time`/`end_time` — as long as the actual nonzero
content agrees and the extra range on either side is genuinely all zero.

```python
def signals_equal(a, b, tolerance=1e-9):
    """
    Return True if a and b represent the same signal, treating any time index
    outside either signal's stored range as 0 (which get_value_at_time already
    does for you). Return False if any time index in the combined range of
    both signals has a value that differs by more than `tolerance`.
    """
    # TODO
```

Test: `a = make_signal(0, 2, [1,2,3])` and `b = make_signal(-2, 4, [0,0,1,2,3,0,0])`
should compare equal, even though their ranges are completely different. Also test a
case where they should NOT be equal (change one value) and confirm your function
correctly returns `False`.

---

## Q3 — Design an echo / reverb effect (15 marks, ~15 min)

A simple digital echo effect works by adding delayed, quieter copies of a "dry" (clean)
signal on top of itself — e.g. an echo every 3 samples, each half as loud as the one
before it. This is nothing more than convolving the dry signal with a specially
designed impulse response.

```python
def make_echo_impulse_response(period, num_echoes, decay):
    """
    Build and return a DiscreteSignal impulse response h that represents:
      - the original ("dry") sample at time 0 with strength 1.0
      - an echo at time period, with strength decay^1
      - an echo at time 2*period, with strength decay^2
      - ... up to num_echoes echoes total (echoes 1 through num_echoes - 1;
        the "0th echo" is just the original dry sample at time 0)
    Everything else in h should be 0.
    """
    # TODO
```

Build `h = make_echo_impulse_response(period=3, num_echoes=4, decay=0.5)`, apply it to
a short "dry" signal `make_signal(0, 2, [1.0, 0.5, -0.3])` using `LTISystem`, and print
the resulting "wet" (with-echo) signal. Explain in a comment why building the whole
effect as a single impulse response (rather than manually adding 4 separately shifted
copies in your main code) is a cleaner design.

---

## Q4 — Denoise a signal and measure the improvement (15 marks, ~15 min)

A moving-average filter is one of the simplest denoising tools: it blends each sample
with its neighbors, which tends to cancel out random noise while mostly preserving the
underlying trend.

```python
def add_noise(signal, noise_std, seed=None):
    """
    Return a NEW signal with independent Gaussian noise (mean 0, std noise_std)
    added to every sample. You may use numpy.random for generating the noise
    values themselves (that's not "using a convolution function", so it's
    allowed) -- but adding the noise into a DiscreteSignal must be done using
    DiscreteSignal operations (set_value_at_time or similar).
    """
    # TODO

def rms_error(a, b):
    """
    Return sqrt(mean((a[n] - b[n])^2)) over the combined range of a and b --
    a standard way to measure how 'different' two signals are.
    """
    # TODO
```

Build a clean signal of your choice (at least 20 samples, a slowly-varying or
piecewise-flat shape works best), add noise with `noise_std = 0.5`, then apply a
5-point moving average filter to the noisy version.

**Important gotcha to watch for:** a causal moving average filter has a "warm-up"
transient at the very start — `y[0]` only has `x[0]` to work with (there's no `x[-1]`,
`x[-2]`, etc.), so the first `filter_length - 1` output samples are averaged over
fewer real inputs than the rest and can look unusually far from the clean signal. If
you compare RMS error over *every* sample including this transient, the filtered
signal can misleadingly look *worse* than the raw noisy one. Fix this by skipping the
first `filter_length - 1` samples when computing RMS error for a fair comparison (add
a `skip_start` parameter to `rms_error`, or just slice the comparison range).

Print `rms_error(clean, noisy)` and `rms_error(clean, filtered)`, both computed
skipping the warm-up region — the second should be noticeably smaller, proving the
filter helped. State in a print statement what fraction the error was reduced by, and
explain in a comment why the transient happens and why skipping it is the fair thing
to do (not "cheating" — it's excluding a region where the filter fundamentally
doesn't have enough data yet, not cherry-picking favorable data).

---

## Q5 — Change-point / edge detector (15 marks, ~15 min)

The first-difference filter (`h = [1, -1]`) computes `y[n] = x[n] - x[n-1]` — it's large
whenever the signal jumps suddenly, and near-zero when the signal is flat. This is a
basic building block of edge detection.

```python
def detect_change_points(signal, threshold):
    """
    Apply the first-difference filter (h[0]=1, h[1]=-1) to `signal` using
    LTISystem, then return a list of every time index n where
    abs(y[n]) > threshold -- i.e. every place the signal jumped by more than
    the threshold amount between consecutive samples.
    """
    # TODO
```

Build a signal that's mostly flat but has two sudden jumps (e.g. flat at 2.0 from
n=0..9, jumps to 8.0 for n=10..14, drops to 3.0 for n=15..19 — build this with
`make_signal` directly, listing out the values). Run `detect_change_points` with
`threshold = 2.0` and print the detected change points — they should land close to
n=10 and n=15 (the exact index depends on which side of the jump the filter reports
it on — figure out which side it is and explain why in a comment).

---

## Q6 — Polynomial multiplication via convolution (15 marks, ~15 min)

Here's a fact that surprises a lot of people: multiplying two polynomials is
*mathematically identical* to convolving their coefficient lists. If
`p1(x) = 1 + 2x + 3x^2` and `p2(x) = 4 + 5x`, their product's coefficients are exactly
`convolve([1,2,3], [4,5])`.

```python
def multiply_polynomials(coeffs1, coeffs2):
    """
    coeffs1 and coeffs2 are plain Python lists, coeffs[i] being the
    coefficient of x^i (so coeffs1 = [1,2,3] means 1 + 2x + 3x^2).
    Build DiscreteSignal objects out of both (indices 0..len-1), convolve
    them using LTISystem, and return the result as a plain Python list of
    coefficients (use nonzero_samples or just read out signal.values --
    think about which is safer if a middle coefficient happens to be exactly
    zero).
    """
    # TODO
```

Test: `multiply_polynomials([1,2,3], [4,5])` should give `[4, 13, 22, 15]` — verify
this by hand-expanding `(1+2x+3x^2)(4+5x)` on paper first, then check your function's
output matches. Also test `multiply_polynomials([1,-1], [1,1])` — this is
`(1-x)(1+x)`, a classic difference-of-squares — and confirm you get `[1, 0, -1]`.

---

## Q7 — Matched filter: find a hidden pulse in a noisy signal (20 marks, ~20 min)

A "matched filter" is used to detect where a known pattern (a "template") occurs inside
a noisy signal — this is how things like GPS receivers and radar systems find a known
signal buried in noise. The trick: correlate the noisy signal against the template
(convolve with the time-reversed template), and the correlation's **peak** tells you
where the pattern most likely starts.

```python
def time_reverse(signal):
    """Return y such that y[n] = signal[-n]."""
    # TODO

def find_best_match(noisy_signal, template):
    """
    Correlate noisy_signal against template (convolve noisy_signal with the
    time-reversed template, using LTISystem), then return the single time
    index where the correlation result is LARGEST -- that's your best guess
    for where the template starts inside noisy_signal.
    """
    # TODO
```

Build a template `make_signal(0, 2, [1.0, 2.0, 1.0])` (a small triangular pulse).
Build a noisy signal at least 20 samples long filled with small random Gaussian noise
(std 0.2), then add the template's shape on top starting at some time index of your
choice (e.g. n=10) so it's "hidden" in the noise. Run `find_best_match` and print
both the true hidden offset and the detected offset — they should match (or be very
close) despite the noise.

---

## Q8 — Estimate the delay between two signals (20 marks, ~20 min)

Given two versions of the same underlying signal, one of them delayed relative to the
other (e.g. the same sound recorded by two microphones at different distances), you can
recover the delay by finding the peak of their cross-correlation.

```python
def estimate_delay(reference_signal, delayed_signal):
    """
    Correlate delayed_signal against reference_signal (same trick as Q7 --
    time-reverse the reference and convolve). Return the lag at which the
    correlation is maximized -- that lag IS your estimate of how much
    delayed_signal was shifted relative to reference_signal.
    """
    # TODO
```

Build a reference signal with a distinctive shape (not flat -- e.g.
`make_signal(0, 6, [0,1,3,5,3,1,0])`), create a delayed copy using `.shift(4)`, and run
`estimate_delay` on the pair. Print the true delay (4) and your estimated delay --
they should match exactly for a clean, noise-free case like this.

---

## Q9 — Signal restoration: undo a known blur (25 marks, ~25 min, hard)

This is the reverse of normal filtering: you're given a known blur/filter `h` and the
*already-blurred* output `y`, and you need to recover the original, unblurred input
`x`. This is a real technique (deconvolution) used in image and audio restoration.

```python
def restore_signal(h, y, x_length):
    """
    h: a known causal DiscreteSignal impulse response with h.start_time == 0
       and h[0] != 0.
    y: the known blurred output, y = x * h for some unknown x.
    x_length: how many samples x has, starting at n=0.
    Recover and return x using the rearranged convolution formula:
        x[0] = y[0] / h[0]
        x[n] = ( y[n] - sum_{k=0}^{n-1} x[k]*h[n-k] ) / h[0]     for n > 0
    """
    # TODO
```

Test it properly: pick some "true" `x` of your choice, pick a causal `h` with
`h[0] != 0`, generate `y = LTISystem(h).output(x_true)`, then call
`restore_signal(h, y, len(x_true))` and confirm the recovered signal matches `x_true`
almost exactly (max difference near 0).

Explain in a comment: if `h[0]` were 0 instead of nonzero, would this method still
work? Why or why not?

---

## Q10 — When does a recursive filter equal a (truncated) FIR filter? (25 marks, ~25 min, hard)

A "leaky integrator" is a simple recursive (IIR) filter: `y[n] = x[n] + r*y[n-1]`,
with `0 < r < 1`. If you feed it a single impulse (`x = [1, 0, 0, 0, ...]`), the output
you get back is, sample by sample, the filter's own (infinite, in theory) impulse
response — and it turns out to just be `r^n` for each `n`.

```python
def system_leaky_integrator(input_signal, r):
    """
    Implement y[n] = x[n] + r*y[n-1] with initial rest (y = 0 before
    input_signal.start_time), using a single forward loop.
    """
    # TODO

def truncated_geometric_impulse_response(r, length):
    """
    Build and return a DiscreteSignal h over [0, length-1] where
    h[n] = r^n for each n -- a finite (FIR) APPROXIMATION of the leaky
    integrator's true (infinite) impulse response.
    """
    # TODO
```

Feed `system_leaky_integrator` a unit impulse (`x = make_signal(0, 0, [1.0])`, extended
implicitly to whatever range you compute over — think about how far you need to run the
recursion to get, say, 5 meaningful output samples) with `r = 0.5`, and separately build
`truncated_geometric_impulse_response(0.5, 5)`. Compare the two — they should match
almost exactly for the first several samples. Then explain in a comment: since the true
impulse response of this recursive system is technically infinite (`r^n` never actually
reaches exactly 0 for any finite n), what does "truncating" it to a finite FIR filter
actually cost you in accuracy, and does that cost get better or worse as `r` gets
closer to 1?

---

## Q11 — Sparse signal compression (10 marks, ~10 min)

Real signals are often mostly zero (silence, empty space, no event). Instead of storing
every single sample, you can store only the nonzero ones plus the total length — a
simple form of compression.

```python
def compress(signal):
    """
    Return a compact representation: a tuple (start_time, end_time,
    list_of_nonzero_samples) where list_of_nonzero_samples is exactly what
    signal.nonzero_samples() gives you. This should be MUCH smaller than
    storing every value if the signal is mostly zero.
    """
    # TODO

def decompress(compressed):
    """
    Given a tuple in the format produced by compress(), rebuild and return
    the original DiscreteSignal exactly (a full DiscreteSignal object with
    every sample restored, zeros included).
    """
    # TODO
```

Build a sparse signal (e.g. `DiscreteSignal(0, 99)` with only 4 or 5 nonzero samples set
by hand), compress it, decompress it, and confirm (using your `signals_equal` from Q2,
or `max_absolute_difference`) that the round trip is lossless. Print how many numbers
your compressed representation actually stores versus the signal's full length of 100.

---

## Q12 — Naive convolution vs. the offline's sparse-aware convolution (15 marks, ~15 min)

Your `LTISystem.output_by_superposition` only loops over the input's *nonzero* samples
(via `nonzero_samples()`), which is efficient when the input is sparse (mostly zero).
A naive implementation would loop over *every* time index in the input's range,
regardless of whether it's zero, which wastes work on a sparse signal.

```python
def naive_convolve(x, h):
    """
    Compute x * h using a straightforward double loop over EVERY time index
    in x's range and EVERY time index in h's range (no skipping zeros,
    no use of nonzero_samples). Build the output the direct way:
    for every n_x in x.times(), for every n_h in h.times(),
    add x[n_x]*h[n_h] into the output at time n_x + n_h.
    Return the result as a DiscreteSignal over the correct output range.
    """
    # TODO
```

Build a **sparse** input signal spanning a wide range (e.g. `DiscreteSignal(0, 199)`
with only 3 nonzero samples set by hand) and a small `h` (e.g. 5 samples). Time how
long `naive_convolve(x, h)` takes versus `LTISystem(h).output_by_superposition(x)`
(use Python's `time.perf_counter()`). Confirm both give the identical result (max
difference ~0), then print both timings and state in a print statement which one was
faster and why that makes sense given how sparse `x` is.
