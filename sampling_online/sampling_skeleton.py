"""Copy-ready NumPy helpers for sampling and signal-reconstruction problems."""

from collections.abc import Callable
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

__all__ = [
    # --- time axes, sample counts, array sizing ---------------------------
    "make_time_axis",           # uniform sample times over [start, stop)
    "sample_signal",            # evaluate a vectorized signal on that grid
    "sample_count",             # how many samples a duration holds at fs
    "frequency_resolution",     # fs/N: the spacing of DFT bins, in Hz
    "next_power_of_two",        # smallest 2**k >= n, for radix-2 FFT sizes
    "pad_to_power_of_two",      # zero-pad a sequence up to that length
    "zero_pad",                 # pad either side with a constant value
    # --- building signals -------------------------------------------------
    "sinusoid",                 # A*cos(2*pi*f*t + phase)
    "sum_of_tones",             # several sinusoids added, one call
    "unit_step",                # u(t - at), with a choosable value at the jump
    "ramp",                     # slope*(t - at) for t >= at, else 0
    "rectangular_pulse",        # unit-height rect on a chosen interval
    "triangular_pulse",         # unit-height triangle, zero outside
    "square_wave",              # +/-A square wave with a duty cycle
    "unit_impulse",             # discrete delta: one 1 in a row of zeros
    "add_noise",                # add Gaussian noise at a target SNR in dB
    # --- rates, Nyquist, aliasing ------------------------------------------
    "nyquist_rate",             # 2*f_max, in samples per second
    "meets_nyquist",            # does fs satisfy the lecture's condition?
    "alias_frequency",          # fold ONE frequency into [0, fs/2]
    "alias_frequencies",        # the same fold, vectorized over an array
    "is_aliased",               # per-tone: was this frequency moved?
    "describe_sampling",        # dict summary of fs vs a list of tones
    "decimate",                 # keep every Mth sample (no anti-alias filter)
    # --- spectra ------------------------------------------------------------
    "magnitude_spectrum",       # one-sided amplitude spectrum, real input
    "magnitude_spectrum_db",    # the same, in decibels, with a floor
    "two_sided_spectrum",       # full complex DFT, fftshifted to -fs/2..fs/2
    "dominant_frequencies",     # the strongest bins, strongest first
    "spectrum_replicas",        # the (1/T) sum_k X(f - k*fs) copy picture
    "ideal_lowpass_filter",     # zero every bin above a cutoff, then inverse
    # --- convolution ---------------------------------------------------------
    "circular_convolve",        # periodic convolution, via the FFT
    "linear_convolve_via_fft",  # np.convolve's answer, computed by FFT
    "cross_correlate",          # correlation and its lag axis, for delays
    # --- reconstruction -------------------------------------------------------
    "sinc_reconstruct",         # sum x[n] sinc((t - nT)/T) at any output times
    "sinc_kernel",              # a truncated sinc sampled on the fine grid
    "zero_order_hold",          # hold x[n] over [nT, (n+1)T)
    "zoh_kernel",               # the ZOH rect sampled on the fine grid
    "upsample",                 # zero-stuff: the impulse-train array model
    "interpolate_by_convolution",  # zero-stuff, convolve, re-align
    "zoh_frequency_response",   # T*exp(-j*pi*f*T)*sinc(f*T)
    "zoh_compensation_gain",    # 1/|sinc(f*T)|: the gain that undoes droop
    "reconstruction_error",     # max-abs, RMS and SNR against a reference
    # --- plotting ---------------------------------------------------------------
    "plot_signal",              # one curve, or one stem plot
    "plot_sampling",            # dense reference curve plus sample stems
    "plot_spectrum",            # magnitude spectrum, optional fs/2 and fs marks
    "plot_reconstruction",      # reference, reconstruction and samples together
    "plot_zoh",                 # the held staircase, drawn with step(where=post)
]


def _as_1d_array(values: ArrayLike, name: str = "values") -> NDArray[Any]:
    """Return *values* as a one-dimensional NumPy array."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def make_time_axis(
    start: float,
    stop: float,
    fs: float,
    endpoint: bool = False,
) -> NDArray[np.float64]:
    """Create uniformly spaced sample times in seconds.

    The interval is ``[start, stop)`` by default. If ``endpoint`` is true,
    ``stop`` is included when it lies on the sampling grid.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    if stop <= start:
        raise ValueError("stop must be greater than start")

    span_in_samples = (stop - start) * fs
    count = int(np.ceil(span_in_samples - 1e-12))
    if endpoint and np.isclose(span_in_samples, round(span_in_samples)):
        count += 1
    return start + np.arange(count, dtype=float) / fs


def sample_signal(
    signal: Callable[[NDArray[np.float64]], ArrayLike | float],
    fs: float,
    start: float,
    stop: float,
    endpoint: bool = False,
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Evaluate a vectorized signal callable on a uniform sampling grid."""
    if not callable(signal):
        raise ValueError("signal must be callable")

    times = make_time_axis(start, stop, fs, endpoint=endpoint)
    values = np.asarray(signal(times))
    if values.ndim == 0:
        values = np.full(times.shape, values)
    if values.shape != times.shape:
        raise ValueError("signal must return one value per sample time")
    return times, values


def zero_pad(
    values: ArrayLike,
    left: int = 0,
    right: int = 0,
    value: float | complex = 0.0,
) -> NDArray[Any]:
    """Pad a one-dimensional sequence on either side with a constant value."""
    array = _as_1d_array(values)
    if not isinstance(left, (int, np.integer)) or not isinstance(
        right, (int, np.integer)
    ):
        raise ValueError("padding widths must be integers")
    if left < 0 or right < 0:
        raise ValueError("padding widths must be nonnegative")
    return np.pad(array, (left, right), constant_values=value)


def unit_step(
    t: ArrayLike,
    at: float = 0.0,
    value_at_zero: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate a unit step whose jump occurs at time ``at``."""
    times = np.asarray(t, dtype=float)
    return np.where(
        times < at,
        0.0,
        np.where(times > at, 1.0, value_at_zero),
    )


def rectangular_pulse(
    t: ArrayLike,
    start: float,
    stop: float,
    closed: str = "left",
) -> NDArray[np.float64]:
    """Return a unit-height rectangular pulse on the selected interval."""
    if stop <= start:
        raise ValueError("stop must be greater than start")
    if closed not in {"left", "right", "both", "neither"}:
        raise ValueError("closed must be 'left', 'right', 'both', or 'neither'")

    times = np.asarray(t, dtype=float)
    left_mask = times >= start if closed in {"left", "both"} else times > start
    right_mask = times <= stop if closed in {"right", "both"} else times < stop
    return (left_mask & right_mask).astype(float)


def nyquist_rate(max_frequency_hz: float) -> float:
    """Return twice the highest signal frequency, in samples per second."""
    if max_frequency_hz < 0:
        raise ValueError("max_frequency_hz must be nonnegative")
    return 2.0 * max_frequency_hz


def meets_nyquist(
    fs: float,
    max_frequency_hz: float,
    strict: bool = True,
) -> bool:
    """Check whether ``fs`` satisfies the Nyquist sampling condition."""
    if fs <= 0:
        raise ValueError("fs must be positive")
    rate = nyquist_rate(max_frequency_hz)
    return bool(fs > rate if strict else fs >= rate)


def alias_frequency(frequency_hz: float, fs: float) -> float:
    """Fold a real sinusoid into the nonnegative interval ``[0, fs/2]``."""
    if fs <= 0:
        raise ValueError("fs must be positive")
    wrapped = (abs(frequency_hz) + fs / 2.0) % fs - fs / 2.0
    return float(abs(wrapped))


def magnitude_spectrum(
    values: ArrayLike,
    fs: float,
    remove_mean: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a one-sided amplitude spectrum for real, uniform samples."""
    array = _as_1d_array(values)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if fs <= 0:
        raise ValueError("fs must be positive")

    working = array - np.mean(array) if remove_mean else array
    spectrum = np.fft.rfft(working)
    magnitude = np.abs(spectrum) * (2.0 / array.size)
    magnitude[0] /= 2.0
    if array.size % 2 == 0:
        magnitude[-1] /= 2.0
    frequencies = np.fft.rfftfreq(array.size, d=1.0 / fs)
    return frequencies, magnitude


def _validate_sample_pairs(
    sample_times: ArrayLike,
    sample_values: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Validate paired one-dimensional sample times and values."""
    times = _as_1d_array(sample_times, "sample_times").astype(float)
    values = _as_1d_array(sample_values, "sample_values")
    if times.size == 0:
        raise ValueError("sample_times must not be empty")
    if times.size != values.size:
        raise ValueError("sample_times and sample_values must have equal length")
    if np.any(np.diff(times) <= 0):
        raise ValueError("sample_times must be strictly increasing")
    return times, values


def sinc_reconstruct(
    sample_times: ArrayLike,
    sample_values: ArrayLike,
    output_times: ArrayLike,
    sample_period: float | None = None,
) -> NDArray[Any]:
    """Reconstruct uniform samples using shifted normalized sinc functions."""
    times, values = _validate_sample_pairs(sample_times, sample_values)
    output = np.asarray(output_times, dtype=float)

    if sample_period is None:
        if times.size < 2:
            raise ValueError(
                "at least two sample times are needed to infer sample_period"
            )
        differences = np.diff(times)
        sample_period = float(differences[0])
        if not np.allclose(
            differences,
            sample_period,
            rtol=1e-7,
            atol=1e-12,
        ):
            raise ValueError("sample_times must be uniformly spaced")
    elif sample_period <= 0:
        raise ValueError("sample_period must be positive")

    kernel = np.sinc((output[..., np.newaxis] - times) / sample_period)
    return kernel @ values


def zero_order_hold(
    sample_times: ArrayLike,
    sample_values: ArrayLike,
    output_times: ArrayLike,
    fill_value: float | complex = np.nan,
) -> NDArray[Any]:
    """Hold each sample over its left-closed interval until the next sample."""
    times, values = _validate_sample_pairs(sample_times, sample_values)
    output = np.asarray(output_times, dtype=float)
    indices = np.searchsorted(times, output, side="right") - 1
    valid = (output >= times[0]) & (output <= times[-1])
    dtype = np.result_type(values.dtype, np.asarray(fill_value).dtype, float)
    held = np.full(output.shape, fill_value, dtype=dtype)
    held[valid] = values[indices[valid]]
    return held


def zoh_frequency_response(
    frequencies_hz: ArrayLike,
    sample_period: float,
) -> NDArray[np.complex128]:
    """Evaluate ``T exp(-j*pi*f*T) sinc(f*T)`` for a zero-order hold."""
    if sample_period <= 0:
        raise ValueError("sample_period must be positive")
    frequencies = np.asarray(frequencies_hz, dtype=float)
    return (
        sample_period
        * np.exp(-1j * np.pi * frequencies * sample_period)
        * np.sinc(frequencies * sample_period)
    )


def plot_signal(
    t: ArrayLike,
    values: ArrayLike,
    *,
    ax: Axes | None = None,
    discrete: bool = False,
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
    grid: bool = True,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Plot a one-dimensional signal as a curve or a discrete stem plot."""
    times = _as_1d_array(t, "t")
    array = _as_1d_array(values)
    if times.size != array.size:
        raise ValueError("t and values must have equal length")

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    if discrete:
        ax.stem(times, array, label=label, **style)
    else:
        ax.plot(times, array, label=label, **style)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(grid, alpha=0.3)
    if label:
        ax.legend()
    return figure, ax


def plot_sampling(
    reference_times: ArrayLike,
    reference_values: ArrayLike,
    sample_times: ArrayLike,
    sample_values: ArrayLike,
    *,
    ax: Axes | None = None,
    title: str = "Continuous signal and samples",
) -> tuple[Figure, Axes]:
    """Overlay discrete samples on a dense reference curve."""
    reference_time_array = _as_1d_array(reference_times, "reference_times")
    reference_value_array = _as_1d_array(reference_values, "reference_values")
    sample_time_array = _as_1d_array(sample_times, "sample_times")
    sample_value_array = _as_1d_array(sample_values, "sample_values")
    if reference_time_array.size != reference_value_array.size:
        raise ValueError(
            "reference_times and reference_values must have equal length"
        )
    if sample_time_array.size != sample_value_array.size:
        raise ValueError("sample_times and sample_values must have equal length")

    figure, ax = plot_signal(
        reference_time_array,
        reference_value_array,
        ax=ax,
        label="reference",
        title=title,
    )
    ax.stem(
        sample_time_array,
        sample_value_array,
        linefmt="C1-",
        markerfmt="C1o",
        basefmt="k-",
        label="samples",
    )
    ax.legend()
    return figure, ax


def two_sided_spectrum(
    values: ArrayLike,
    fs: float,
    shift: bool = True,
    remove_mean: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return the full two-sided DFT and its frequency axis in hertz.

    Uses ``np.fft.fft`` / ``np.fft.fftfreq``. With ``shift`` true the output is
    reordered by ``np.fft.fftshift`` so frequencies run from ``-fs/2`` up to
    just below ``+fs/2``, which is the ordering needed to draw the lecture's
    spectrum-replication picture.
    """
    array = _as_1d_array(values)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if fs <= 0:
        raise ValueError("fs must be positive")

    working = array - np.mean(array) if remove_mean else array
    spectrum = np.fft.fft(working)
    frequencies = np.fft.fftfreq(array.size, d=1.0 / fs)
    if shift:
        spectrum = np.fft.fftshift(spectrum)
        frequencies = np.fft.fftshift(frequencies)
    return frequencies, spectrum


def spectrum_replicas(
    frequencies_hz: ArrayLike,
    magnitude: ArrayLike,
    fs: float,
    copies: int = 2,
    points: int = 2001,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the shifted-copy picture produced by impulse-train sampling.

    ``frequencies_hz``/``magnitude`` describe one baseband spectrum. Each copy
    ``k`` is that shape translated to ``k*fs`` and scaled by ``1/T = fs``, as in
    ``Xp = (1/T) sum_k X(f - k*fs)``. Returns ``(grid, stack, total)`` where
    ``stack`` has one row per copy and ``total`` is their sum; overlap in
    ``total`` is exactly aliasing.
    """
    baseband_f = _as_1d_array(frequencies_hz, "frequencies_hz").astype(float)
    baseband_mag = _as_1d_array(magnitude, "magnitude").astype(float)
    if baseband_f.size != baseband_mag.size:
        raise ValueError("frequencies_hz and magnitude must have equal length")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if copies < 0:
        raise ValueError("copies must be nonnegative")

    order = np.argsort(baseband_f)
    baseband_f = baseband_f[order]
    baseband_mag = baseband_mag[order]

    span = (copies + 0.5) * fs
    grid = np.linspace(-span, span, points)
    offsets = np.arange(-copies, copies + 1, dtype=float) * fs
    stack = fs * np.array(
        [
            np.interp(grid - offset, baseband_f, baseband_mag, left=0.0, right=0.0)
            for offset in offsets
        ]
    )
    return grid, stack, stack.sum(axis=0)


def upsample(values: ArrayLike, factor: int) -> NDArray[Any]:
    """Zero-stuff a sequence: keep ``x[n]``, insert ``factor-1`` zeros after it.

    This is the array model of the impulse train ``xp(t) = sum x(nT) d(t - nT)``
    placed on a grid ``factor`` times finer. Convolving the result with a
    reconstruction kernel performs interpolation.
    """
    array = _as_1d_array(values)
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    stuffed = np.zeros(array.size * int(factor), dtype=array.dtype)
    stuffed[:: int(factor)] = array
    return stuffed


def zoh_kernel(factor: int) -> NDArray[np.float64]:
    """Return the zero-order-hold kernel: ``factor`` ones, i.e. a unit rect."""
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    return np.ones(int(factor), dtype=float)


def sinc_kernel(factor: int, half_width: int = 10) -> NDArray[np.float64]:
    """Return a truncated normalized sinc kernel for convolution interpolation.

    The kernel spans ``half_width`` sample periods on each side and is sampled
    on the upsampled grid, so its centre index (its delay) is
    ``half_width * factor``.
    """
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if not isinstance(half_width, (int, np.integer)) or half_width < 1:
        raise ValueError("half_width must be a positive integer")
    reach = int(half_width) * int(factor)
    return np.sinc(np.arange(-reach, reach + 1) / float(factor))


def interpolate_by_convolution(
    values: ArrayLike,
    factor: int,
    kernel: ArrayLike,
    delay: int | None = None,
) -> NDArray[Any]:
    """Reconstruct by zero-stuffing and convolving with a kernel.

    Equivalent to ``np.convolve(upsample(values, factor), kernel)`` trimmed so
    that output index ``n*factor`` lines up with input sample ``n``. ``delay``
    is the kernel's centre index; it defaults to 0 for a causal kernel such as
    the ZOH rect and should be ``half_width*factor`` for :func:`sinc_kernel`.
    """
    stuffed = upsample(values, factor)
    taps = _as_1d_array(kernel, "kernel")
    if taps.size == 0:
        raise ValueError("kernel must not be empty")
    if delay is None:
        delay = 0
    if not isinstance(delay, (int, np.integer)) or delay < 0:
        raise ValueError("delay must be a nonnegative integer")

    full = np.convolve(stuffed, taps)
    return full[int(delay) : int(delay) + stuffed.size]


def plot_spectrum(
    frequencies_hz: ArrayLike,
    magnitude: ArrayLike,
    *,
    ax: Axes | None = None,
    discrete: bool = False,
    label: str | None = None,
    title: str | None = None,
    fs: float | None = None,
    **style: Any,
) -> tuple[Figure, Axes]:
    """Plot a magnitude spectrum, optionally marking ``fs/2`` and ``fs``."""
    figure, ax = plot_signal(
        frequencies_hz,
        magnitude,
        ax=ax,
        discrete=discrete,
        label=label,
        title=title,
        xlabel="Frequency (Hz)",
        ylabel="Magnitude",
        **style,
    )
    if fs is not None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        ax.axvline(fs / 2.0, color="C3", linestyle="--", label="fs/2")
        ax.axvline(fs, color="C2", linestyle=":", label="fs")
        ax.legend()
    return figure, ax


def sample_count(duration: float, fs: float, endpoint: bool = False) -> int:
    """Return how many samples a duration of ``duration`` seconds holds at ``fs``.

    Matches :func:`make_time_axis` exactly, so ``sample_count(D, fs)`` is always
    ``make_time_axis(0, D, fs).size``. The half-open interval gives ``D*fs``
    samples; ``endpoint`` adds the closing sample when it lands on the grid.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    return int(make_time_axis(0.0, duration, fs, endpoint=endpoint).size)


def frequency_resolution(n: int, fs: float) -> float:
    """Return the DFT bin spacing ``fs/n`` in hertz.

    This equals ``1/(n*T)``, the reciprocal of the record duration. Two tones
    closer together than this cannot be told apart, and zero padding does not
    change it — only a longer record does.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if fs <= 0:
        raise ValueError("fs must be positive")
    return float(fs) / int(n)


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``.

    Radix-2 FFTs are fastest at these lengths, and it is the usual choice when
    zero padding a record before ``np.fft.fft``.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    return 1 << (int(n) - 1).bit_length()


def pad_to_power_of_two(values: ArrayLike) -> NDArray[Any]:
    """Zero-pad a sequence on the right up to the next power-of-two length.

    Padding interpolates the frequency grid more finely. It does not improve
    resolution and it does not change the sampling rate.
    """
    array = _as_1d_array(values)
    if array.size == 0:
        raise ValueError("values must not be empty")
    return zero_pad(array, right=next_power_of_two(array.size) - array.size)


def sinusoid(
    t: ArrayLike,
    frequency_hz: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate ``amplitude * cos(2*pi*frequency_hz*t + phase)``.

    Pass ``phase=-np.pi/2`` for a sine. The frequency is in hertz; if a question
    gives ``cos(omega*t)`` use ``frequency_hz = omega/(2*np.pi)``.
    """
    times = np.asarray(t, dtype=float)
    return amplitude * np.cos(2.0 * np.pi * frequency_hz * times + phase)


def sum_of_tones(
    t: ArrayLike,
    frequencies_hz: ArrayLike,
    amplitudes: ArrayLike | None = None,
    phases: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Add several cosines evaluated on the same time array.

    Broadcasts one row per tone and sums them, so no Python loop runs over the
    time instants. The highest entry of ``frequencies_hz`` is the ``f_max`` that
    sets the Nyquist rate.
    """
    times = np.asarray(t, dtype=float)
    frequencies = _as_1d_array(frequencies_hz, "frequencies_hz").astype(float)
    if frequencies.size == 0:
        raise ValueError("frequencies_hz must not be empty")

    if amplitudes is None:
        amplitude_array = np.ones_like(frequencies)
    else:
        amplitude_array = _as_1d_array(amplitudes, "amplitudes").astype(float)
    if phases is None:
        phase_array = np.zeros_like(frequencies)
    else:
        phase_array = _as_1d_array(phases, "phases").astype(float)

    if amplitude_array.size != frequencies.size or phase_array.size != frequencies.size:
        raise ValueError("amplitudes and phases must match frequencies_hz in length")

    tones = amplitude_array[:, np.newaxis] * np.cos(
        2.0 * np.pi * frequencies[:, np.newaxis] * times[np.newaxis, :]
        + phase_array[:, np.newaxis]
    )
    return tones.sum(axis=0)


def ramp(t: ArrayLike, at: float = 0.0, slope: float = 1.0) -> NDArray[np.float64]:
    """Return ``slope*(t - at)`` for ``t >= at`` and zero before it."""
    times = np.asarray(t, dtype=float)
    return np.where(times >= at, slope * (times - at), 0.0)


def triangular_pulse(
    t: ArrayLike,
    center: float = 0.0,
    half_width: float = 1.0,
) -> NDArray[np.float64]:
    """Return a unit-height triangle peaking at ``center``, zero outside."""
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    times = np.asarray(t, dtype=float)
    return np.clip(1.0 - np.abs(times - center) / half_width, 0.0, None)


def square_wave(
    t: ArrayLike,
    frequency_hz: float,
    duty: float = 0.5,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Return a ``+/-amplitude`` square wave of the given frequency.

    A square wave has harmonics at every odd multiple of ``frequency_hz``, so
    its bandwidth is unbounded and no finite ``fs`` avoids aliasing entirely.
    It is the standard example of a signal the theorem cannot rescue.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if not 0.0 < duty < 1.0:
        raise ValueError("duty must lie strictly between 0 and 1")
    times = np.asarray(t, dtype=float)
    within_period = np.mod(times * frequency_hz, 1.0)
    return np.where(within_period < duty, amplitude, -amplitude)


def unit_impulse(length: int, index: int = 0) -> NDArray[np.float64]:
    """Return a discrete delta: ``length`` zeros with a single 1 at ``index``.

    This is the discrete ``delta[n - index]``, not a Dirac impulse. Negative
    indices count from the end, as elsewhere in NumPy.
    """
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError("length must be a positive integer")
    if not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer")
    if not -int(length) <= int(index) < int(length):
        raise ValueError("index must lie inside the array")
    values = np.zeros(int(length), dtype=float)
    values[int(index)] = 1.0
    return values


def add_noise(
    values: ArrayLike,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Add zero-mean Gaussian noise at a target signal-to-noise ratio in dB.

    Noise power is set to ``signal_power / 10**(snr_db/10)`` from the measured
    mean square of ``values``. Pass ``rng=np.random.default_rng(0)`` when a
    grader must reproduce the figure.
    """
    array = _as_1d_array(values).astype(float)
    if array.size == 0:
        raise ValueError("values must not be empty")
    generator = np.random.default_rng() if rng is None else rng

    signal_power = float(np.mean(array**2))
    if signal_power == 0.0:
        raise ValueError("values must not be all zero")
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return array + generator.normal(0.0, np.sqrt(noise_power), size=array.shape)


def alias_frequencies(frequencies_hz: ArrayLike, fs: float) -> NDArray[np.float64]:
    """Fold a whole array of real-sinusoid frequencies into ``[0, fs/2]``.

    The vectorized twin of :func:`alias_frequency`. Every tone above ``fs/2``
    reappears at the folded frequency and is indistinguishable from a real tone
    there once sampling has happened.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    frequencies = np.abs(np.asarray(frequencies_hz, dtype=float))
    wrapped = np.mod(frequencies + fs / 2.0, fs) - fs / 2.0
    return np.abs(wrapped)


def is_aliased(
    frequencies_hz: ArrayLike,
    fs: float,
    strict: bool = True,
) -> NDArray[np.bool_]:
    """Return, per tone, whether sampling at ``fs`` moves that frequency.

    A tone is aliased when its folded frequency differs from its own. With
    ``strict`` true the critical case ``f == fs/2`` is also flagged, because the
    lecture's condition is ``fs > 2*f_max`` and samples of a sine at exactly
    ``fs/2`` can all land on zero.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    frequencies = np.abs(np.asarray(frequencies_hz, dtype=float))
    moved = ~np.isclose(alias_frequencies(frequencies, fs), frequencies)
    if strict:
        return moved | np.isclose(frequencies, fs / 2.0)
    return moved


def describe_sampling(
    fs: float,
    frequencies_hz: ArrayLike,
    strict: bool = True,
) -> dict[str, Any]:
    """Summarize one sampling rate against a list of tones, as a dictionary.

    Returns ``f_max``, ``nyquist_rate``, the sampling period ``T``, whether the
    rate is adequate, and the aliased tones as ``(original, folded)`` pairs --
    the answer to most "state your conclusion" parts in one object.
    """
    frequencies = _as_1d_array(frequencies_hz, "frequencies_hz").astype(float)
    if frequencies.size == 0:
        raise ValueError("frequencies_hz must not be empty")
    if fs <= 0:
        raise ValueError("fs must be positive")

    f_max = float(np.max(np.abs(frequencies)))
    flagged = is_aliased(frequencies, fs, strict=strict)
    folded = alias_frequencies(frequencies, fs)
    return {
        "fs": float(fs),
        "T": 1.0 / float(fs),
        "f_max": f_max,
        "nyquist_rate": nyquist_rate(f_max),
        "satisfies_nyquist": meets_nyquist(fs, f_max, strict=strict),
        "aliased": [
            (float(original), float(alias))
            for original, alias, bad in zip(frequencies, folded, flagged)
            if bad
        ],
    }


def decimate(values: ArrayLike, factor: int) -> NDArray[Any]:
    """Keep every ``factor``-th sample, dropping the rest.

    This is naive decimation: it lowers the effective rate to ``fs/factor`` and
    will alias anything above the new ``fs/(2*factor)``. Low-pass filter first
    (see :func:`ideal_lowpass_filter`) if the signal is not already band-limited
    for the new rate.
    """
    array = _as_1d_array(values)
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    return array[:: int(factor)]


def magnitude_spectrum_db(
    values: ArrayLike,
    fs: float,
    floor_db: float = -120.0,
    remove_mean: bool = False,
    reference: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a one-sided amplitude spectrum in decibels.

    ``20*log10(amplitude/reference)``, clipped at ``floor_db`` so that exact
    zeros do not become ``-inf``. ``reference`` defaults to the largest
    amplitude present, which puts the tallest peak at 0 dB.
    """
    frequencies, amplitude = magnitude_spectrum(values, fs, remove_mean=remove_mean)
    scale = float(np.max(amplitude)) if reference is None else float(reference)
    if scale <= 0:
        raise ValueError("reference must be positive")
    with np.errstate(divide="ignore"):
        decibels = 20.0 * np.log10(amplitude / scale)
    return frequencies, np.maximum(decibels, floor_db)


def dominant_frequencies(
    frequencies_hz: ArrayLike,
    magnitude: ArrayLike,
    count: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ``count`` strongest bins of a spectrum, strongest first.

    Sorts by magnitude with ``np.argsort`` and reorders both arrays by the same
    indices. Bins next to a peak are also large, so on a leaky spectrum the top
    few entries may all belong to one tone.
    """
    frequencies = _as_1d_array(frequencies_hz, "frequencies_hz").astype(float)
    amplitude = _as_1d_array(magnitude, "magnitude").astype(float)
    if frequencies.size != amplitude.size:
        raise ValueError("frequencies_hz and magnitude must have equal length")
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")

    take = min(int(count), amplitude.size)
    order = np.argsort(amplitude)[::-1][:take]
    return frequencies[order], amplitude[order]


def ideal_lowpass_filter(
    values: ArrayLike,
    fs: float,
    cutoff_hz: float,
) -> NDArray[np.float64]:
    """Zero every DFT bin above ``cutoff_hz`` and transform back.

    The numerical stand-in for the lecture's ideal reconstruction filter
    ``H(j omega) = T`` inside the band and 0 outside. It is a brick wall in
    frequency, so it rings in time (Gibbs); that is the price of an ideal
    filter, not a coding mistake.
    """
    array = _as_1d_array(values).astype(float)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0.0 < cutoff_hz <= fs / 2.0:
        raise ValueError("cutoff_hz must lie in (0, fs/2]")

    spectrum = np.fft.rfft(array)
    frequencies = np.fft.rfftfreq(array.size, d=1.0 / fs)
    spectrum[frequencies > cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, n=array.size)


def circular_convolve(
    a: ArrayLike,
    b: ArrayLike,
    n: int | None = None,
) -> NDArray[Any]:
    """Convolve two sequences periodically, using the FFT.

    Computes ``ifft(fft(a, n) * fft(b, n))``: multiplication of DFTs is
    *circular* convolution, so the tail wraps around instead of extending the
    array. ``n`` defaults to the longer input. Use
    :func:`linear_convolve_via_fft` when you want ``np.convolve``'s answer.
    """
    first = _as_1d_array(a, "a")
    second = _as_1d_array(b, "b")
    if first.size == 0 or second.size == 0:
        raise ValueError("a and b must not be empty")

    length = max(first.size, second.size) if n is None else int(n)
    if length < max(first.size, second.size):
        raise ValueError("n must be at least as long as the longer input")

    product = np.fft.fft(first, length) * np.fft.fft(second, length)
    result = np.fft.ifft(product)
    if np.isrealobj(first) and np.isrealobj(second):
        return np.real(result)
    return result


def linear_convolve_via_fft(a: ArrayLike, b: ArrayLike) -> NDArray[Any]:
    """Compute ``np.convolve(a, b)`` through the FFT.

    Circular convolution equals linear convolution once both inputs are padded
    to at least ``len(a) + len(b) - 1``, which stops the wrap-around. This
    function pads to the next power of two and trims, so the result matches
    ``np.convolve(a, b, "full")`` to floating-point accuracy.
    """
    first = _as_1d_array(a, "a")
    second = _as_1d_array(b, "b")
    if first.size == 0 or second.size == 0:
        raise ValueError("a and b must not be empty")

    needed = first.size + second.size - 1
    return circular_convolve(first, second, next_power_of_two(needed))[:needed]


def cross_correlate(
    a: ArrayLike,
    b: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[Any]]:
    """Return ``(lags, correlation)`` for the full cross-correlation of a with b.

    Unlike convolution, correlation does not flip the second sequence. The lag
    at the largest magnitude is how far ``a`` is delayed relative to ``b``, so
    ``lags[np.argmax(np.abs(correlation))]`` estimates a delay in samples.
    """
    first = _as_1d_array(a, "a")
    second = _as_1d_array(b, "b")
    if first.size == 0 or second.size == 0:
        raise ValueError("a and b must not be empty")

    correlation = np.correlate(first, second, mode="full")
    lags = np.arange(-(second.size - 1), first.size, dtype=np.int64)
    return lags, correlation


def zoh_compensation_gain(
    frequencies_hz: ArrayLike,
    sample_period: float,
) -> NDArray[np.float64]:
    """Return ``1/|sinc(f*T)|``, the gain that flattens zero-order-hold droop.

    The ZOH magnitude response is ``T*|sinc(f*T)|``, normalized here to unit DC
    gain. The response is zero at every multiple of ``fs``, so the compensation
    is infinite there; those entries come back as ``np.inf`` and cannot be
    corrected by any filter.
    """
    if sample_period <= 0:
        raise ValueError("sample_period must be positive")
    frequencies = np.asarray(frequencies_hz, dtype=float)
    response = np.abs(np.sinc(frequencies * sample_period))
    return np.where(response > 1e-12, 1.0 / np.maximum(response, 1e-12), np.inf)


def reconstruction_error(
    reference: ArrayLike,
    reconstructed: ArrayLike,
) -> dict[str, float]:
    """Compare a reconstruction against a reference on the same time grid.

    Returns ``max_abs``, ``rms`` and ``snr_db``. Ideal sinc reconstruction from
    a finite record is worst near the ends, so compare an interior slice if you
    want to quote the error away from the edges.
    """
    expected = _as_1d_array(reference, "reference").astype(float)
    actual = _as_1d_array(reconstructed, "reconstructed").astype(float)
    if expected.size != actual.size:
        raise ValueError("reference and reconstructed must have equal length")
    if expected.size == 0:
        raise ValueError("reference must not be empty")

    difference = actual - expected
    error_power = float(np.mean(difference**2))
    signal_power = float(np.mean(expected**2))
    if error_power == 0.0:
        snr_db = float("inf")
    elif signal_power == 0.0:
        snr_db = float("-inf")
    else:
        snr_db = 10.0 * np.log10(signal_power / error_power)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(error_power)),
        "snr_db": float(snr_db),
    }


def plot_reconstruction(
    output_times: ArrayLike,
    reconstructed: ArrayLike,
    sample_times: ArrayLike,
    sample_values: ArrayLike,
    *,
    reference: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str = "Reconstruction from samples",
    label: str = "reconstruction",
) -> tuple[Figure, Axes]:
    """Draw a reconstruction, its samples, and optionally the original curve."""
    times = _as_1d_array(output_times, "output_times")
    values = _as_1d_array(reconstructed, "reconstructed")
    if times.size != values.size:
        raise ValueError("output_times and reconstructed must have equal length")

    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure

    if reference is not None:
        reference_values = _as_1d_array(reference, "reference")
        if reference_values.size != times.size:
            raise ValueError("reference must match output_times in length")
        ax.plot(times, reference_values, color="C0", label="original")

    ax.plot(times, values, "--", color="C1", label=label)
    ax.stem(
        _as_1d_array(sample_times, "sample_times"),
        _as_1d_array(sample_values, "sample_values"),
        linefmt="C2-",
        markerfmt="C2o",
        basefmt="k-",
        label="samples",
    )
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title=title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return figure, ax


def plot_zoh(
    output_times: ArrayLike,
    held_values: ArrayLike,
    sample_times: ArrayLike,
    sample_values: ArrayLike,
    *,
    reference: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str = "Zero-order hold",
) -> tuple[Figure, Axes]:
    """Draw a zero-order hold as a staircase, with its samples.

    Uses ``step(where="post")``, which holds each value until the next output
    time -- the ``[nT, (n+1)T)`` convention. ``where="pre"`` would shift the
    whole staircase back by one sample period.
    """
    times = _as_1d_array(output_times, "output_times")
    values = _as_1d_array(held_values, "held_values")
    if times.size != values.size:
        raise ValueError("output_times and held_values must have equal length")

    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure

    if reference is not None:
        reference_values = _as_1d_array(reference, "reference")
        if reference_values.size != times.size:
            raise ValueError("reference must match output_times in length")
        ax.plot(times, reference_values, color="C0", label="original")

    ax.step(times, values, where="post", color="C1", label="ZOH")
    ax.stem(
        _as_1d_array(sample_times, "sample_times"),
        _as_1d_array(sample_values, "sample_values"),
        linefmt="C2-",
        markerfmt="C2o",
        basefmt="k-",
        label="samples",
    )
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title=title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return figure, ax
