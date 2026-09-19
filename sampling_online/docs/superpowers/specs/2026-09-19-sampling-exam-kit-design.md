# Sampling Exam Kit Design

## Purpose

Create a copy-friendly exam reference for a Signals and Linear Systems coding test. The kit will help a student quickly recognize a sampling problem, select the relevant NumPy operations, adapt a reliable code template, and check the result with Matplotlib.

The deliverables are a Markdown guide and a standalone Python skeleton. They are study and coding aids, not a replacement for the lecture derivations.

## Source and Syllabus Boundary

The source is `C:\Users\Raufur\Downloads\Lecture 5 - Sampling.pdf`.

Included material is PDF pages 1-31, through the complete zero-order-hold discussion:

- Continuous-time signals represented by uniform samples
- Ambiguity of sampled representations
- Sampling as multiplication by an impulse train
- Multiplication/convolution duality needed for the sampling argument
- Spectrum of an impulse train
- Spectral replication caused by sampling
- Nyquist-Shannon sampling theorem
- Oversampling, undersampling, and aliasing
- Ideal low-pass reconstruction and sinc interpolation
- Zero-order hold, its rectangular impulse response, and its frequency response

Excluded material begins on PDF page 32:

- Linear interpolation / first-order hold
- Comparison of reconstruction filters beyond ZOH
- DFT-focused material and periodic-signal derivations

FFT code may appear only as a practical visualization tool for signals and aliasing. The guide must label it as a numerical spectrum estimate rather than teach the later DFT syllabus.

## Deliverables

### `sampling_exam_guide.md`

A self-contained guide optimized for searching and copying during practice. It will contain:

1. A syllabus map and notation table, including conversions among `T`, `fs`, `omega_s`, `f_max`, and `omega_M`.
2. A compact NumPy reference covering array creation, shapes, dtypes, indexing, slicing, masks, Boolean operations, broadcasting, vectorization, concatenation, padding, reductions, mathematical functions, `np.where`, and useful inspection calls.
3. Matplotlib patterns for continuous-looking plots, discrete sample/stem plots, overlays, subplots, labels, limits, legends, and grids.
4. Signal-building recipes for sinusoids, sums of tones, exponentials, unit steps, rectangular pulses, and piecewise signals.
5. Sampling workflows: choosing a sample rate, constructing the time grid, evaluating a callable signal, comparing dense reference and sampled values, and detecting Nyquist violations.
6. Aliasing workflows: predicting the apparent frequency and demonstrating two continuous sinusoids that produce the same sample sequence.
7. Ideal reconstruction using shifted normalized sinc functions.
8. ZOH reconstruction using repeated values or interval lookup, plus a frequency-response visualization.
9. Reusable problem-solving templates and a checklist of common mistakes.
10. Setter-style practice problems followed by complete solutions and brief reasoning.

Code blocks will be executable after importing NumPy and Matplotlib. Examples will use ordinary frequency in hertz for code while explicitly connecting it to angular frequency in radians per second.

### `sampling_skeleton.py`

A flat, copy-friendly module with no dependencies beyond NumPy and Matplotlib. It will include:

- Constants/import aliases used throughout the guide
- `make_time_axis(start, stop, fs, endpoint=False)`
- `sample_signal(signal, fs, start, stop, endpoint=False)`
- `zero_pad(values, left=0, right=0, value=0.0)`
- `unit_step(t, at=0.0, value_at_zero=1.0)`
- `rectangular_pulse(t, start, stop, closed="left")`
- `nyquist_rate(max_frequency_hz)`
- `meets_nyquist(fs, max_frequency_hz, strict=True)`
- `alias_frequency(frequency_hz, fs)`
- `magnitude_spectrum(values, fs, remove_mean=False)`
- `sinc_reconstruct(sample_times, sample_values, output_times, sample_period=None)`
- `zero_order_hold(sample_times, sample_values, output_times, fill_value=np.nan)`
- `zoh_frequency_response(frequencies_hz, sample_period)`
- `plot_signal(...)`
- `plot_sampling(...)`

Functions will accept one-dimensional numerical arrays, return NumPy arrays, validate conditions that would silently produce misleading results, and avoid unnecessary class-based abstractions.

## Behavioral Decisions

### Time axes

Sampling grids will default to a half-open interval `[start, stop)`. This matches `np.arange`-style sampling and prevents accidentally adding a sample at the end of a duration. Internally, the implementation will calculate the sample count and use `np.arange(count) / fs` to reduce floating-point endpoint surprises.

### Nyquist equality

The lecture states the recovery condition as `fs > 2*f_max`, so `meets_nyquist` will use strict inequality by default. A `strict=False` option will support the common textbook shorthand `fs >= 2*f_max`, with the guide explaining why sampling exactly at the boundary can lose phase-dependent signals.

### Alias frequency

`alias_frequency` will fold a real sinusoidal frequency into the nonnegative Nyquist interval `[0, fs/2]`. It will accept negative input frequencies by using real-sinusoid symmetry.

### Ideal reconstruction

`sinc_reconstruct` will implement

`x_r(t) = sum_n x[n] sinc((t - nT) / T)`.

If no period is supplied, the function will infer it from uniformly spaced sample times. It will reject fewer than two samples, non-increasing times, and nonuniform spacing. The guide will state that finite records cause edge/truncation error even when the underlying ideal theory assumes infinitely many samples.

### Zero-order hold

`zero_order_hold` will hold sample `n` over `[t_n, t_(n+1))`. At the last sample time, it will return the last sample. Times outside the sampled interval will receive `fill_value` rather than silently extrapolate.

The frequency-response helper will evaluate

`H0(f) = T * exp(-j*pi*f*T) * sinc(f*T)`,

which is the hertz-form equivalent of the lecture's angular-frequency formula.

### Plotting

Plotting functions will return `(fig, ax)` and will not call `plt.show()`. This lets a student add titles, limits, or annotations before displaying the figure. The sampling plot will overlay a dense reference curve and sample stems.

## Practice Problem Coverage

The guide will include problems representative of a NumPy/Matplotlib sessional:

1. Create and inspect arrays, slice them, and use masks.
2. Construct and plot a piecewise signal with Boolean masks.
3. Sample a single sinusoid above Nyquist and overlay samples on a dense reference.
4. Determine a sufficient sampling rate for a sum of sinusoids.
5. Demonstrate aliasing and calculate the apparent frequency.
6. Show that two different continuous-time sinusoids have identical samples.
7. Reconstruct a sampled signal with finite sinc interpolation and discuss edge error.
8. Reconstruct a signal with ZOH and compare it visually with the original.
9. Plot ZOH magnitude response and identify droop at higher baseband frequencies.
10. Solve a mixed problem requiring sampling, padding, masking, reconstruction, and plotting.

Solutions will be placed after the full problem set so the questions can be attempted without immediately seeing the answer.

## Error Handling

The skeleton will raise `ValueError` for:

- Nonpositive sample rates or periods
- Stop time not greater than start time
- Non-one-dimensional sample arrays
- Mismatched time/value lengths
- Empty sample arrays where reconstruction is requested
- Nonuniform sample times for sinc reconstruction
- Invalid pulse-boundary modes
- Negative padding sizes

Messages will name the invalid argument and state the expected condition.

## Verification

Automated tests will verify:

- Correct time-axis length and spacing
- Signal callable evaluation on the generated grid
- Left/right zero padding and validation
- Unit-step and rectangular-pulse boundary behavior
- Strict and non-strict Nyquist checks
- Known alias-frequency cases
- Spectrum peak location for a bin-centered sinusoid
- Exact sinc reconstruction at original sample times
- Rejection of nonuniform sample times
- ZOH behavior inside intervals, at sample points, and outside the sample range
- ZOH response at DC and at its first null
- Plot helpers returning Matplotlib objects without showing a window

The final verification will run the full test suite, compile the Python file, execute every Python code block in the guide where practical, and generate representative plots using a noninteractive backend.

## Scope Controls

- No SciPy dependency; normalized sinc is available as `np.sinc`.
- No object-oriented framework or package scaffolding.
- No GUI or interactive widgets.
- No first-order hold, linear interpolation, or later DFT theory.
- No attempt to numerically represent Dirac impulses as ordinary finite-height samples without an explicit explanation.
- No PDF output because Markdown is more useful for copying code in an exam-preparation workflow.
