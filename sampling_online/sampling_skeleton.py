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
