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
    [
        (9.0, 10.0, 1.0),
        (6.0, 10.0, 4.0),
        (14.0, 10.0, 4.0),
        (-6.0, 10.0, 4.0),
    ],
)
def test_alias_frequency_folds_into_nyquist_interval(frequency, fs, expected):
    assert ss.alias_frequency(frequency, fs) == pytest.approx(expected)
