"""Copy-ready NumPy helpers for sampling and signal-reconstruction problems."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
