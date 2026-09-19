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
        sample_times,
        sample_values,
        sample_times,
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
    np.testing.assert_allclose(
        result,
        [-99.0, 10.0, 10.0, 20.0, 20.0, 30.0, -99.0],
    )


def test_zoh_frequency_response_has_dc_gain_T_and_first_null():
    response = ss.zoh_frequency_response([0.0, 4.0], sample_period=0.25)
    assert response[0] == pytest.approx(0.25 + 0.0j)
    assert abs(response[1]) == pytest.approx(0.0, abs=1e-12)


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


def test_two_sided_spectrum_is_ascending_and_conjugate_symmetric():
    fs = 100.0
    t = np.arange(int(fs)) / fs
    f, spectrum = ss.two_sided_spectrum(np.sin(2 * np.pi * 12 * t), fs)
    assert np.all(np.diff(f) > 0)
    peaks = np.abs(f[np.argsort(np.abs(spectrum))[-2:]])
    np.testing.assert_allclose(np.sort(peaks), [12.0, 12.0])

    f_raw, _ = ss.two_sided_spectrum(np.sin(2 * np.pi * 12 * t), fs, shift=False)
    assert f_raw[0] == 0.0


def test_upsample_zero_stuffs_between_samples():
    np.testing.assert_allclose(ss.upsample([1.0, 2.0], 3), [1, 0, 0, 2, 0, 0])


def test_zoh_kernel_convolution_matches_zero_order_hold():
    factor = 8
    samples = np.array([1.0, -2.0, 0.5, 3.0])
    held = ss.interpolate_by_convolution(samples, factor, ss.zoh_kernel(factor))
    np.testing.assert_allclose(held, np.repeat(samples, factor))


def test_sinc_kernel_convolution_is_exact_at_sample_instants():
    factor = 8
    samples = np.array([1.0, -2.0, 0.5, 3.0])
    kernel = ss.sinc_kernel(factor, half_width=12)
    dense = ss.interpolate_by_convolution(
        samples, factor, kernel, delay=12 * factor
    )
    np.testing.assert_allclose(dense[::factor], samples, atol=1e-9)


def test_spectrum_replicas_scales_by_fs_and_overlaps_when_undersampled():
    baseband_f = np.linspace(-4.0, 4.0, 401)
    baseband_mag = np.clip(1.0 - np.abs(baseband_f) / 4.0, 0.0, None)

    grid, stack, total = ss.spectrum_replicas(baseband_f, baseband_mag, fs=20.0)
    assert stack.shape == (5, grid.size)
    np.testing.assert_allclose(total, stack.sum(axis=0))
    np.testing.assert_allclose(total[np.argmin(np.abs(grid))], 20.0)

    grid, stack, total = ss.spectrum_replicas(
        baseband_f, baseband_mag, fs=5.0, copies=1
    )
    overlap = int(np.argmin(np.abs(grid - 2.5)))
    assert total[overlap] > stack[:, overlap].max()


def test_plot_spectrum_marks_nyquist_and_sampling_frequency():
    frequencies = np.linspace(0.0, 50.0, 101)
    _, ax = ss.plot_spectrum(frequencies, np.ones_like(frequencies), fs=40.0)
    assert len(ax.lines) >= 3
