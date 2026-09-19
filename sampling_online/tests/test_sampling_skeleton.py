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


# --- sizing helpers ----------------------------------------------------------


@pytest.mark.parametrize("n, expected", [(1, 1), (2, 2), (3, 4), (5, 8), (64, 64), (65, 128)])
def test_next_power_of_two(n, expected):
    assert ss.next_power_of_two(n) == expected


def test_pad_to_power_of_two_only_adds_zeros_on_the_right():
    padded = ss.pad_to_power_of_two([1.0, 2.0, 3.0])
    np.testing.assert_allclose(padded, [1.0, 2.0, 3.0, 0.0])


def test_sample_count_agrees_with_make_time_axis():
    assert ss.sample_count(1.0, 10.0) == ss.make_time_axis(0.0, 1.0, 10.0).size == 10
    assert ss.sample_count(1.0, 10.0, endpoint=True) == 11


def test_frequency_resolution_is_the_reciprocal_of_record_duration():
    assert ss.frequency_resolution(100, 100.0) == 1.0
    assert ss.frequency_resolution(200, 100.0) == 0.5


# --- signal builders ---------------------------------------------------------


def test_sinusoid_phase_shift_turns_cosine_into_sine():
    t = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(
        ss.sinusoid(t, 3.0, phase=-np.pi / 2),
        np.sin(2 * np.pi * 3.0 * t),
        atol=1e-12,
    )


def test_sum_of_tones_matches_the_explicit_sum():
    t = np.linspace(0.0, 1.0, 101)
    combined = ss.sum_of_tones(t, [3.0, 7.0], amplitudes=[2.0, 0.5])
    expected = 2.0 * np.cos(2 * np.pi * 3.0 * t) + 0.5 * np.cos(2 * np.pi * 7.0 * t)
    np.testing.assert_allclose(combined, expected)


def test_sum_of_tones_rejects_mismatched_amplitudes():
    with pytest.raises(ValueError, match="must match frequencies_hz"):
        ss.sum_of_tones([0.0, 1.0], [3.0, 7.0], amplitudes=[1.0])


def test_ramp_is_zero_before_the_break_and_linear_after():
    t = np.array([-1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(ss.ramp(t, at=0.0, slope=2.0), [0.0, 0.0, 2.0, 4.0])


def test_triangular_pulse_peaks_at_one_and_vanishes_outside():
    t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(ss.triangular_pulse(t), [0.0, 0.0, 1.0, 0.0, 0.0])


def test_square_wave_alternates_and_respects_duty():
    t = np.linspace(0.0, 1.0, 1000, endpoint=False)
    wave = ss.square_wave(t, frequency_hz=1.0, duty=0.25)
    assert set(np.unique(wave)) == {-1.0, 1.0}
    assert np.isclose(np.mean(wave > 0), 0.25, atol=0.01)


def test_unit_impulse_places_a_single_one():
    np.testing.assert_allclose(ss.unit_impulse(4, 2), [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(ss.unit_impulse(4, -1), [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="inside the array"):
        ss.unit_impulse(4, 4)


def test_add_noise_hits_the_requested_snr_and_is_reproducible():
    t = np.linspace(0.0, 1.0, 20000, endpoint=False)
    clean = np.sin(2 * np.pi * 5 * t)
    noisy = ss.add_noise(clean, snr_db=20.0, rng=np.random.default_rng(0))
    measured = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
    assert abs(measured - 20.0) < 0.5

    repeat = ss.add_noise(clean, snr_db=20.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(noisy, repeat)


# --- aliasing ----------------------------------------------------------------


def test_alias_frequencies_matches_the_scalar_version():
    frequencies = np.array([1.0, 9.0, 14.0, 21.0])
    np.testing.assert_allclose(
        ss.alias_frequencies(frequencies, 10.0),
        [ss.alias_frequency(f, 10.0) for f in frequencies],
    )


def test_is_aliased_flags_tones_above_half_the_rate():
    flagged = ss.is_aliased([1.0, 4.0, 5.0, 9.0], fs=10.0)
    np.testing.assert_array_equal(flagged, [False, False, True, True])
    relaxed = ss.is_aliased([5.0], fs=10.0, strict=False)
    np.testing.assert_array_equal(relaxed, [False])


def test_describe_sampling_summarizes_rate_and_aliased_tones():
    summary = ss.describe_sampling(10.0, [1.0, 9.0])
    assert summary["f_max"] == 9.0
    assert summary["nyquist_rate"] == 18.0
    assert summary["satisfies_nyquist"] is False
    assert summary["T"] == 0.1
    assert summary["aliased"] == [(9.0, 1.0)]


def test_decimate_keeps_every_nth_sample():
    np.testing.assert_allclose(ss.decimate(np.arange(6.0), 2), [0.0, 2.0, 4.0])


# --- spectra -----------------------------------------------------------------


def test_magnitude_spectrum_db_puts_the_peak_at_zero_decibels():
    fs = 100.0
    t = np.arange(int(fs)) / fs
    f, decibels = ss.magnitude_spectrum_db(np.sin(2 * np.pi * 12 * t), fs)
    assert np.isclose(decibels.max(), 0.0)
    assert f[np.argmax(decibels)] == 12.0
    assert decibels.min() >= -120.0


def test_dominant_frequencies_returns_strongest_first():
    frequencies = np.array([1.0, 2.0, 3.0])
    magnitude = np.array([0.2, 1.0, 0.5])
    top_f, top_mag = ss.dominant_frequencies(frequencies, magnitude, count=2)
    np.testing.assert_allclose(top_f, [2.0, 3.0])
    np.testing.assert_allclose(top_mag, [1.0, 0.5])


def test_ideal_lowpass_filter_removes_the_out_of_band_tone():
    fs = 200.0
    t = np.arange(int(fs)) / fs
    mixed = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 60 * t)
    filtered = ss.ideal_lowpass_filter(mixed, fs, cutoff_hz=20.0)
    np.testing.assert_allclose(filtered, np.sin(2 * np.pi * 5 * t), atol=1e-9)


def test_ideal_lowpass_filter_rejects_a_cutoff_above_nyquist():
    with pytest.raises(ValueError, match=r"cutoff_hz must lie in \(0, fs/2\]"):
        ss.ideal_lowpass_filter(np.ones(8), 10.0, cutoff_hz=6.0)


# --- convolution -------------------------------------------------------------


def test_circular_convolve_wraps_the_tail_around():
    result = ss.circular_convolve([1.0, 2.0, 3.0], [1.0, 1.0], n=3)
    np.testing.assert_allclose(result, [4.0, 3.0, 5.0])


def test_circular_convolve_equals_linear_when_padded_enough():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 1.0])
    padded = ss.circular_convolve(a, b, n=a.size + b.size - 1)
    np.testing.assert_allclose(padded, np.convolve(a, b), atol=1e-12)


def test_linear_convolve_via_fft_matches_numpy_convolve():
    rng = np.random.default_rng(0)
    a = rng.normal(size=37)
    b = rng.normal(size=11)
    np.testing.assert_allclose(
        ss.linear_convolve_via_fft(a, b), np.convolve(a, b), atol=1e-10
    )


def test_cross_correlate_lag_finds_a_known_delay():
    pattern = np.array([1.0, -1.0, 0.5])
    delayed = np.concatenate([np.zeros(4), pattern, np.zeros(3)])
    lags, correlation = ss.cross_correlate(delayed, pattern)
    assert lags[np.argmax(np.abs(correlation))] == 4
    assert lags.size == correlation.size


# --- reconstruction ----------------------------------------------------------


def test_zoh_compensation_gain_is_one_at_dc_and_infinite_at_the_null():
    T = 0.1
    gains = ss.zoh_compensation_gain([0.0, 1 / T], T)
    assert np.isclose(gains[0], 1.0)
    assert np.isinf(gains[1])


def test_zoh_compensation_gain_undoes_the_droop():
    T = 0.1
    frequencies = np.array([1.0, 3.0])
    response = np.abs(ss.zoh_frequency_response(frequencies, T)) / T
    np.testing.assert_allclose(
        response * ss.zoh_compensation_gain(frequencies, T), 1.0
    )


def test_reconstruction_error_is_zero_for_an_exact_match():
    reference = np.array([1.0, -2.0, 0.5])
    exact = ss.reconstruction_error(reference, reference)
    assert exact["max_abs"] == 0.0
    assert exact["rms"] == 0.0
    assert np.isinf(exact["snr_db"])

    off = ss.reconstruction_error(reference, reference + 0.1)
    assert np.isclose(off["max_abs"], 0.1)
    assert np.isclose(off["rms"], 0.1)
    assert off["snr_db"] < 30.0


# --- plotting ----------------------------------------------------------------


def test_plot_reconstruction_draws_reference_curve_and_samples():
    t = np.linspace(0.0, 1.0, 51)
    _, ax = ss.plot_reconstruction(
        t, np.sin(t), [0.0, 0.5, 1.0], [0.0, 0.5, 0.8], reference=np.cos(t)
    )
    assert len(ax.lines) >= 3
    assert {"original", "reconstruction", "samples"} <= {
        text.get_text() for text in ax.get_legend().get_texts()
    }


def test_plot_zoh_uses_a_post_step_staircase():
    t = np.linspace(0.0, 1.0, 51)
    _, ax = ss.plot_zoh(t, np.zeros_like(t), [0.0, 0.5], [1.0, -1.0])
    assert any(line.get_drawstyle() == "steps-post" for line in ax.lines)
