"""Copy-ready NumPy helpers for sampling and signal-reconstruction problems."""

from collections.abc import Callable
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "alias_frequency",
    "interpolate_by_convolution",
    "magnitude_spectrum",
    "make_time_axis",
    "meets_nyquist",
    "nyquist_rate",
    "plot_sampling",
    "plot_signal",
    "plot_spectrum",
    "rectangular_pulse",
    "sample_signal",
    "sinc_kernel",
    "sinc_reconstruct",
    "spectrum_replicas",
    "two_sided_spectrum",
    "unit_step",
    "upsample",
    "zero_order_hold",
    "zero_pad",
    "zoh_frequency_response",
    "zoh_kernel",
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
