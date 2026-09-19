# Sampling Exam Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a copy-friendly Markdown sampling guide and a tested NumPy/Matplotlib skeleton covering the lecture through zero-order hold.

**Architecture:** A single flat Python module exposes small numerical and plotting helpers that students can copy independently. A separate Markdown guide teaches the relevant NumPy operations, sampling concepts, problem templates, and worked practice solutions while importing the same helpers in larger examples. Pytest tests pin down numerical boundaries and keep the teaching code trustworthy.

**Tech Stack:** Python 3, NumPy, Matplotlib, pytest, Markdown

**Spec:** `docs/superpowers/specs/2026-09-19-sampling-exam-kit-design.md`

## Global Constraints

- Limit theory to PDF pages 1-31, ending after the complete zero-order-hold frequency response.
- Do not teach linear interpolation, first-order hold, or later DFT theory.
- Use only NumPy and Matplotlib in the student-facing skeleton.
- Use hertz in executable examples and explicitly map hertz to angular frequency.
- Use half-open sampling intervals by default.
- Treat the lecture's Nyquist condition as strict inequality by default.
- Do not call `plt.show()` from plotting helpers.
- Keep all public helpers flat and independently copyable; do not add classes or package scaffolding.

## Review Focus

- Floating-point stop times: `make_time_axis` must not accidentally include the stop value when `endpoint=False`.
- Nyquist boundary: strict mode rejects `fs == 2*f_max`, while non-strict mode accepts it.
- Finite sinc records: reconstruction must be exact at supplied sample times even though off-grid edge error remains.
- ZOH boundaries: every sample owns the interval starting at its timestamp; values outside the recorded interval use `fill_value`.
- FFT interpretation: the guide must call it a numerical visualization tool, not part of the excluded DFT theory.

---

## File Structure

- Create `sampling_skeleton.py`: all copy-ready numerical and plotting helpers.
- Create `tests/test_sampling_skeleton.py`: behavioral tests for every helper.
- Create `sampling_exam_guide.md`: syllabus reference, NumPy/Matplotlib recipes, templates, and practice solutions.

### Task 1: Core Array, Signal, Sampling, and Nyquist Helpers

**Files:**
- Create: `sampling_skeleton.py`
- Create: `tests/test_sampling_skeleton.py`

**Interfaces:**
- Consumes: NumPy arrays, scalar sample rates, and callables that accept a NumPy time array.
- Produces: `make_time_axis`, `sample_signal`, `zero_pad`, `unit_step`, `rectangular_pulse`, `nyquist_rate`, `meets_nyquist`, and `alias_frequency`.

- [ ] **Step 1: Write failing tests for the core helpers**

Create `tests/test_sampling_skeleton.py` with:

```python
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import sampling_skeleton as ss


def test_make_time_axis_uses_half_open_interval_and_requested_spacing():
    t = ss.make_time_axis(0.0, 1.0, fs=4.0)
    np.testing.assert_allclose(t, [0.0, 0.25, 0.5, 0.75])


def test_make_time_axis_can_include_exact_endpoint():
    t = ss.make_time_axis(0.0, 1.0, fs=4.0, endpoint=True)
    np.testing.assert_allclose(t, [0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize("fs", [0.0, -2.0])
def test_make_time_axis_rejects_nonpositive_sample_rate(fs):
    with pytest.raises(ValueError, match="fs must be positive"):
        ss.make_time_axis(0.0, 1.0, fs)


def test_sample_signal_evaluates_callable_on_sampling_grid():
    t, x = ss.sample_signal(lambda time: 2.0 * time, 4.0, 0.0, 1.0)
    np.testing.assert_allclose(t, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0, 1.5])


def test_zero_pad_supports_left_right_and_fill_value():
    result = ss.zero_pad([1, 2], left=1, right=2, value=-1)
    np.testing.assert_array_equal(result, [-1, 1, 2, -1, -1])


def test_zero_pad_rejects_negative_width():
    with pytest.raises(ValueError, match="nonnegative"):
        ss.zero_pad([1, 2], left=-1)


def test_unit_step_supports_selected_value_at_jump():
    t = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(ss.unit_step(t), [0.0, 1.0, 1.0])
    np.testing.assert_allclose(
        ss.unit_step(t, value_at_zero=0.5), [0.0, 0.5, 1.0]
    )


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("left", [0.0, 1.0, 1.0, 0.0]),
        ("right", [0.0, 0.0, 1.0, 1.0]),
        ("both", [0.0, 1.0, 1.0, 1.0]),
        ("neither", [0.0, 0.0, 1.0, 0.0]),
    ],
)
def test_rectangular_pulse_boundary_modes(closed, expected):
    t = np.array([-1.0, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        ss.rectangular_pulse(t, 0.0, 1.0, closed=closed), expected
    )


def test_nyquist_helpers_distinguish_strict_boundary():
    assert ss.nyquist_rate(30.0) == 60.0
    assert not ss.meets_nyquist(60.0, 30.0)
    assert ss.meets_nyquist(60.0, 30.0, strict=False)
    assert ss.meets_nyquist(61.0, 30.0)


@pytest.mark.parametrize(
    ("frequency", "fs", "expected"),
    [(9.0, 10.0, 1.0), (6.0, 10.0, 4.0), (14.0, 10.0, 4.0), (-6.0, 10.0, 4.0)],
)
def test_alias_frequency_folds_into_nyquist_interval(frequency, fs, expected):
    assert ss.alias_frequency(frequency, fs) == pytest.approx(expected)
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist**

Run:

```powershell
python -m pytest tests/test_sampling_skeleton.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'sampling_skeleton'`.

- [ ] **Step 3: Implement the minimal core helpers**

Create `sampling_skeleton.py` with imports, `_as_1d_array`, and the eight public helpers. Use these algorithms:

```python
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _as_1d_array(values, name="values"):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def make_time_axis(start, stop, fs, endpoint=False):
    if fs <= 0:
        raise ValueError("fs must be positive")
    if stop <= start:
        raise ValueError("stop must be greater than start")
    span_in_samples = (stop - start) * fs
    count = int(np.ceil(span_in_samples - 1e-12))
    if endpoint and np.isclose(span_in_samples, round(span_in_samples)):
        count += 1
    return start + np.arange(count, dtype=float) / fs


def sample_signal(signal, fs, start, stop, endpoint=False):
    if not callable(signal):
        raise ValueError("signal must be callable")
    t = make_time_axis(start, stop, fs, endpoint=endpoint)
    values = np.asarray(signal(t))
    if values.ndim == 0:
        values = np.full(t.shape, values)
    if values.shape != t.shape:
        raise ValueError("signal must return one value per sample time")
    return t, values


def zero_pad(values, left=0, right=0, value=0.0):
    values = _as_1d_array(values)
    if not isinstance(left, (int, np.integer)) or not isinstance(
        right, (int, np.integer)
    ):
        raise ValueError("padding widths must be integers")
    if left < 0 or right < 0:
        raise ValueError("padding widths must be nonnegative")
    return np.pad(values, (left, right), constant_values=value)


def unit_step(t, at=0.0, value_at_zero=1.0):
    t = np.asarray(t, dtype=float)
    return np.where(t < at, 0.0, np.where(t > at, 1.0, value_at_zero))


def rectangular_pulse(t, start, stop, closed="left"):
    if stop <= start:
        raise ValueError("stop must be greater than start")
    if closed not in {"left", "right", "both", "neither"}:
        raise ValueError("closed must be 'left', 'right', 'both', or 'neither'")
    t = np.asarray(t, dtype=float)
    left = t >= start if closed in {"left", "both"} else t > start
    right = t <= stop if closed in {"right", "both"} else t < stop
    return (left & right).astype(float)


def nyquist_rate(max_frequency_hz):
    if max_frequency_hz < 0:
        raise ValueError("max_frequency_hz must be nonnegative")
    return 2.0 * max_frequency_hz


def meets_nyquist(fs, max_frequency_hz, strict=True):
    if fs <= 0:
        raise ValueError("fs must be positive")
    rate = nyquist_rate(max_frequency_hz)
    return bool(fs > rate if strict else fs >= rate)


def alias_frequency(frequency_hz, fs):
    if fs <= 0:
        raise ValueError("fs must be positive")
    wrapped = (abs(frequency_hz) + fs / 2.0) % fs - fs / 2.0
    return float(abs(wrapped))
```

Add concise docstrings that state units, interval conventions, parameters, and returns without changing these behaviors.

- [ ] **Step 4: Run the core tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_sampling_skeleton.py -q
```

Expected: all tests in Task 1 pass.

- [ ] **Step 5: Commit the core helpers**

```powershell
git add sampling_skeleton.py tests/test_sampling_skeleton.py
git commit -m "feat: add core sampling helpers"
```

### Task 2: Spectrum, Ideal Reconstruction, and Zero-Order Hold

**Files:**
- Modify: `sampling_skeleton.py`
- Modify: `tests/test_sampling_skeleton.py`

**Interfaces:**
- Consumes: uniformly sampled values or explicit sample times and values.
- Produces: `magnitude_spectrum`, `sinc_reconstruct`, `zero_order_hold`, and `zoh_frequency_response`.

- [ ] **Step 1: Append failing numerical tests**

Append:

```python
def test_magnitude_spectrum_finds_bin_centered_tone():
    fs = 32.0
    t = ss.make_time_axis(0.0, 1.0, fs)
    x = np.cos(2.0 * np.pi * 5.0 * t)
    frequencies, magnitude = ss.magnitude_spectrum(x, fs)
    assert frequencies[np.argmax(magnitude)] == pytest.approx(5.0)
    assert np.max(magnitude) == pytest.approx(1.0)


def test_sinc_reconstruction_is_exact_at_sample_times():
    sample_times = np.arange(5, dtype=float) * 0.25
    sample_values = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    reconstructed = ss.sinc_reconstruct(
        sample_times, sample_values, sample_times
    )
    np.testing.assert_allclose(reconstructed, sample_values, atol=1e-12)


def test_sinc_reconstruction_rejects_nonuniform_times():
    with pytest.raises(ValueError, match="uniformly spaced"):
        ss.sinc_reconstruct([0.0, 0.1, 0.25], [1.0, 2.0, 3.0], [0.1])


def test_zero_order_hold_owns_left_closed_intervals_and_fills_outside():
    result = ss.zero_order_hold(
        [0.0, 1.0, 2.0],
        [10.0, 20.0, 30.0],
        [-0.1, 0.0, 0.9, 1.0, 1.9, 2.0, 2.1],
        fill_value=-99.0,
    )
    np.testing.assert_allclose(result, [-99.0, 10.0, 10.0, 20.0, 20.0, 30.0, -99.0])


def test_zoh_frequency_response_has_dc_gain_T_and_first_null():
    response = ss.zoh_frequency_response([0.0, 4.0], sample_period=0.25)
    assert response[0] == pytest.approx(0.25 + 0.0j)
    assert abs(response[1]) == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run the new tests and verify missing-function failures**

Run:

```powershell
python -m pytest tests/test_sampling_skeleton.py -q
```

Expected: failures report missing `magnitude_spectrum`, `sinc_reconstruct`, `zero_order_hold`, and `zoh_frequency_response`.

- [ ] **Step 3: Implement numerical spectrum and reconstruction helpers**

Append these implementations with concise docstrings:

```python
def magnitude_spectrum(values, fs, remove_mean=False):
    values = _as_1d_array(values)
    if values.size == 0:
        raise ValueError("values must not be empty")
    if fs <= 0:
        raise ValueError("fs must be positive")
    working = values - np.mean(values) if remove_mean else values
    spectrum = np.fft.rfft(working)
    magnitude = np.abs(spectrum) * (2.0 / values.size)
    magnitude[0] /= 2.0
    if values.size % 2 == 0:
        magnitude[-1] /= 2.0
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / fs)
    return frequencies, magnitude


def _validate_sample_pairs(sample_times, sample_values):
    times = _as_1d_array(sample_times, "sample_times").astype(float)
    values = _as_1d_array(sample_values, "sample_values")
    if times.size == 0:
        raise ValueError("sample_times must not be empty")
    if times.size != values.size:
        raise ValueError("sample_times and sample_values must have equal length")
    if np.any(np.diff(times) <= 0):
        raise ValueError("sample_times must be strictly increasing")
    return times, values


def sinc_reconstruct(sample_times, sample_values, output_times, sample_period=None):
    times, values = _validate_sample_pairs(sample_times, sample_values)
    output = np.asarray(output_times, dtype=float)
    if sample_period is None:
        if times.size < 2:
            raise ValueError("at least two sample times are needed to infer sample_period")
        differences = np.diff(times)
        sample_period = float(differences[0])
        if not np.allclose(differences, sample_period, rtol=1e-7, atol=1e-12):
            raise ValueError("sample_times must be uniformly spaced")
    elif sample_period <= 0:
        raise ValueError("sample_period must be positive")
    kernel = np.sinc((output[..., np.newaxis] - times) / sample_period)
    return kernel @ values


def zero_order_hold(sample_times, sample_values, output_times, fill_value=np.nan):
    times, values = _validate_sample_pairs(sample_times, sample_values)
    output = np.asarray(output_times, dtype=float)
    indices = np.searchsorted(times, output, side="right") - 1
    valid = (output >= times[0]) & (output <= times[-1])
    dtype = np.result_type(values.dtype, np.asarray(fill_value).dtype, float)
    held = np.full(output.shape, fill_value, dtype=dtype)
    held[valid] = values[indices[valid]]
    return held


def zoh_frequency_response(frequencies_hz, sample_period):
    if sample_period <= 0:
        raise ValueError("sample_period must be positive")
    frequencies = np.asarray(frequencies_hz, dtype=float)
    return (
        sample_period
        * np.exp(-1j * np.pi * frequencies * sample_period)
        * np.sinc(frequencies * sample_period)
    )
```

- [ ] **Step 4: Run the numerical tests and full suite**

Run:

```powershell
python -m pytest tests/test_sampling_skeleton.py -q
```

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit the reconstruction helpers**

```powershell
git add sampling_skeleton.py tests/test_sampling_skeleton.py
git commit -m "feat: add sampling reconstruction helpers"
```

### Task 3: Copy-Friendly Plotting Helpers

**Files:**
- Modify: `sampling_skeleton.py`
- Modify: `tests/test_sampling_skeleton.py`

**Interfaces:**
- Consumes: one-dimensional time/value arrays and optional existing Matplotlib axes.
- Produces: `plot_signal` and `plot_sampling`, each returning `(Figure, Axes)`.

- [ ] **Step 1: Append failing plotting tests**

Append:

```python
def test_plot_signal_returns_objects_and_draws_line_or_stems():
    fig_line, ax_line = ss.plot_signal([0, 1], [1, 2], label="x(t)")
    fig_stem, ax_stem = ss.plot_signal([0, 1], [1, 2], discrete=True)
    assert fig_line is ax_line.figure
    assert len(ax_line.lines) == 1
    assert fig_stem is ax_stem.figure
    assert len(ax_stem.containers) >= 1


def test_plot_sampling_overlays_reference_and_sample_stems():
    fig, ax = ss.plot_sampling(
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    )
    assert fig is ax.figure
    assert len(ax.lines) >= 1
    assert len(ax.containers) >= 1
```

- [ ] **Step 2: Run the plotting tests and verify missing-function failures**

Run:

```powershell
python -m pytest tests/test_sampling_skeleton.py -q
```

Expected: the two new tests fail because the plotting helpers are absent.

- [ ] **Step 3: Implement the plotting helpers**

Append implementations equivalent to:

```python
def plot_signal(
    t,
    values,
    *,
    ax=None,
    discrete=False,
    label=None,
    title=None,
    xlabel="Time (s)",
    ylabel="Amplitude",
    grid=True,
    **style,
):
    t = _as_1d_array(t, "t")
    values = _as_1d_array(values)
    if t.size != values.size:
        raise ValueError("t and values must have equal length")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if discrete:
        ax.stem(t, values, label=label, **style)
    else:
        ax.plot(t, values, label=label, **style)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(grid, alpha=0.3)
    if label:
        ax.legend()
    return fig, ax


def plot_sampling(
    reference_times,
    reference_values,
    sample_times,
    sample_values,
    *,
    ax=None,
    title="Continuous signal and samples",
):
    reference_times = _as_1d_array(reference_times, "reference_times")
    reference_values = _as_1d_array(reference_values, "reference_values")
    sample_times = _as_1d_array(sample_times, "sample_times")
    sample_values = _as_1d_array(sample_values, "sample_values")
    if reference_times.size != reference_values.size:
        raise ValueError("reference_times and reference_values must have equal length")
    if sample_times.size != sample_values.size:
        raise ValueError("sample_times and sample_values must have equal length")
    fig, ax = plot_signal(
        reference_times,
        reference_values,
        ax=ax,
        label="reference",
        title=title,
    )
    ax.stem(sample_times, sample_values, linefmt="C1-", markerfmt="C1o", basefmt="k-", label="samples")
    ax.legend()
    return fig, ax
```

Add docstrings and an `__all__` list containing only the public helper names.

- [ ] **Step 4: Run tests and compile the module**

Run:

```powershell
python -m pytest tests/test_sampling_skeleton.py -q
python -m py_compile sampling_skeleton.py
```

Expected: all tests pass and compilation exits with status 0.

- [ ] **Step 5: Commit the plotting helpers**

```powershell
git add sampling_skeleton.py tests/test_sampling_skeleton.py
git commit -m "feat: add signal plotting helpers"
```

### Task 4: Sampling Exam Guide and Worked Practice Set

**Files:**
- Create: `sampling_exam_guide.md`

**Interfaces:**
- Consumes: the public API in `sampling_skeleton.py` and the approved syllabus boundary.
- Produces: a standalone, searchable Markdown reference whose code blocks are copy-ready.

- [ ] **Step 1: Write the guide structure and notation reference**

Create these top-level sections in order:

```markdown
# Sampling Coding Test Guide

## 1. Syllabus Boundary and How to Use This Guide
## 2. Formula and Notation Sheet
## 3. NumPy Essentials
## 4. Matplotlib Essentials
## 5. Building Signals with Arrays and Masks
## 6. Uniform Sampling Workflow
## 7. Sampling Theorem and Aliasing
## 8. Ideal Sinc Reconstruction
## 9. Zero-Order Hold
## 10. Copy-Ready Problem Templates
## 11. Common Mistakes and Debugging Checklist
## 12. Practice Problems
## 13. Complete Solutions
```

In the formula sheet, explicitly include:

```text
T = 1/fs
omega_s = 2*pi*fs = 2*pi/T
omega_M = 2*pi*f_max
Nyquist condition from the lecture: fs > 2*f_max
xp(t) = sum_n x(nT) delta(t-nT)
Xp(j omega) = (1/T) sum_k X(j(omega-k omega_s))
xr(t) = sum_n x[n] sinc((t-nT)/T)
h0(t) = 1 on [0,T), otherwise 0
H0(f) = T exp(-j*pi*f*T) sinc(f*T)
```

- [ ] **Step 2: Add the NumPy and Matplotlib quick references**

Cover executable examples for:

- `np.array`, `np.arange`, `np.linspace`, `np.zeros`, `np.ones`, `np.full`, and `np.zeros_like`
- `.shape`, `.ndim`, `.size`, `.dtype`, `.copy()`, `.astype()`
- Indexing, negative indices, slices, strides, and assignment through masks
- Comparisons, `&`, `|`, `~`, parentheses around Boolean expressions, and `np.where`
- Broadcasting with row/column examples
- `np.concatenate`, `np.stack`, `np.pad`, `np.diff`, `np.sum`, `np.mean`, `np.max`, `np.argmax`, and `np.allclose`
- `np.sin`, `np.cos`, `np.exp`, `np.abs`, `np.pi`, `np.sinc`, and `np.fft` only as a spectrum-visualization tool
- Line plots, stems, overlays, subplots, legends, grids, axis limits, and figure saving

Every subsection must include one minimal snippet and one sampling-related snippet.

- [ ] **Step 3: Add conceptual explanations and reusable templates**

Explain and demonstrate:

- The distinction between a dense numerical reference curve and an actual continuous-time signal
- `N = duration * fs` for half-open intervals and how endpoint choices alter sample count
- Extracting `f_max` from a sum of tones
- Why the lecture uses strict `fs > 2*f_max`
- Frequency folding with `alias_frequency`
- Identical sample sequences for tones separated by integer multiples of `fs`
- Spectrum replication as conceptual shifted copies, without pretending finite arrays are Dirac impulses
- Finite sinc reconstruction and its edge/truncation limitation
- ZOH interval behavior and passband droop

Include copy-ready templates for signal definition, sampling/overlay plotting, piecewise masks, spectrum visualization, sinc reconstruction, and ZOH reconstruction.

- [ ] **Step 4: Add ten practice problems before their solutions**

Use the ten problem categories from the specification. Each problem must specify numerical values and a required output. The complete solutions section must provide runnable Python, the expected numerical conclusion where applicable, and one or two sentences connecting the result to theory.

Required anchor results include:

- A `30 Hz` maximum tone has strict Nyquist requirement `fs > 60 Hz`.
- Sampling a `9 Hz` sinusoid at `10 Hz` produces a `1 Hz` alias.
- `sin(2*pi*3*t)` and `sin(2*pi*13*t)` have equal samples at `fs=10 Hz`.
- A ZOH with `T=0.1 s` has DC gain `0.1` and first magnitude null at `10 Hz`.

- [ ] **Step 5: Check guide scope and code references**

Run:

```powershell
rg -n "linear interpolation|first-order hold|FOH|DFT|IDFT" sampling_exam_guide.md
rg -n "make_time_axis|sample_signal|zero_pad|sinc_reconstruct|zero_order_hold|zoh_frequency_response|plot_signal|plot_sampling" sampling_exam_guide.md
```

Expected: any `DFT` mention says it is outside scope or labels FFT as a visualization tool; no FOH teaching appears; every major public helper is referenced.

- [ ] **Step 6: Commit the guide**

```powershell
git add sampling_exam_guide.md
git commit -m "docs: add sampling coding test guide"
```

### Task 5: Final Cross-Artifact Verification

**Files:**
- Verify: `sampling_skeleton.py`
- Verify: `tests/test_sampling_skeleton.py`
- Verify: `sampling_exam_guide.md`

**Interfaces:**
- Consumes: all completed deliverables.
- Produces: fresh evidence that the kit imports, tests, and representative examples execute.

- [ ] **Step 1: Run the complete automated test suite**

```powershell
python -m pytest -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Compile and import the skeleton**

```powershell
python -m py_compile sampling_skeleton.py
python -c "import sampling_skeleton as ss; print(sorted(ss.__all__))"
```

Expected: both commands exit 0 and print all public helpers.

- [ ] **Step 3: Execute a representative end-to-end sampling example**

Run with Matplotlib's noninteractive backend:

```powershell
$env:MPLBACKEND='Agg'
python -c "import numpy as np; import sampling_skeleton as ss; f=lambda t: np.sin(2*np.pi*3*t); ts,xs=ss.sample_signal(f,10,0,1); td=ss.make_time_axis(0,1,1000); xr=ss.sinc_reconstruct(ts,xs,td); xz=ss.zero_order_hold(ts,xs,td); fig,ax=ss.plot_sampling(td,f(td),ts,xs); assert len(ts)==10; assert np.allclose(ss.sinc_reconstruct(ts,xs,ts),xs); assert ss.alias_frequency(13,10)==3; print('end-to-end example passed')"
```

Expected: `end-to-end example passed`.

- [ ] **Step 4: Verify guide completeness against the approved specification**

Check that all 13 guide sections exist, all 10 unsolved problems precede all solutions, every solution includes runnable code, and all lecture topics through ZOH appear. Confirm no later-syllabus derivation was introduced.

- [ ] **Step 5: Inspect the final diff and working tree**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors. Report any unrelated pre-existing changes without modifying them.
