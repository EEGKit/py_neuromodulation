import numpy as np
import pandas as pd
import pytest

from py_neuromodulation import (
    nm_settings,
    nm_stream_offline,
    nm_define_nmchannels
)


def get_example_settings(test_arr: np.ndarray) -> tuple[dict, pd.DataFrame]:
    settings = nm_settings.set_settings_fast_compute(
        nm_settings.get_default_settings()
    )

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(test_arr)

    return settings, nm_channels


def test_different_sampling_rate_100Hz():
    sampling_rate_features = 100

    arr_test = np.random.random([2, 1020])
    settings, nm_channels = get_example_settings(arr_test)

    settings["sampling_rate_features_hz"] = sampling_rate_features
    stream = nm_stream_offline.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df = stream.run(arr_test)

    # check the difference between time points

    assert np.diff(df["time"].iloc[:2]) / 1000 == (1 / sampling_rate_features)


def test_different_sampling_rate_10Hz():
    sampling_rate_features = 10

    arr_test = np.random.random([2, 1200])
    settings, nm_channels = get_example_settings(arr_test)

    settings["sampling_rate_features_hz"] = sampling_rate_features
    stream = nm_stream_offline.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df = stream.run(arr_test)

    # check the difference between time points

    assert np.diff(df["time"].iloc[:2]) / 1000 == (1 / sampling_rate_features)


def test_different_sampling_rate_1Hz():
    sampling_rate_features = 1

    arr_test = np.random.random([2, 3000])
    settings, nm_channels = get_example_settings(arr_test)

    settings["sampling_rate_features_hz"] = sampling_rate_features
    stream = nm_stream_offline.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df = stream.run(arr_test)

    # check the difference between time points

    assert np.diff(df["time"].iloc[:2]) / 1000 == (1 / sampling_rate_features)


def test_different_sampling_rate_0DOT1Hz():
    sampling_rate_features = 0.1

    arr_test = np.random.random([2, 30000])
    settings, nm_channels = get_example_settings(arr_test)

    settings["sampling_rate_features_hz"] = sampling_rate_features
    stream = nm_stream_offline.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df = stream.run(arr_test)

    # check the difference between time points

    assert np.diff(df["time"].iloc[:2]) / 1000 == (1 / sampling_rate_features)


def test_wrong_initalization_of_segment_length_features_ms_and_osc_window_length():
    segment_length_features_ms = 800

    arr_test = np.random.random([2, 1200])
    settings, nm_channels = get_example_settings(arr_test)

    settings["segment_length_features_ms"] = 800
    settings["fft_settings"]["windowlength_ms"] = 1000

    with pytest.raises(Exception):
        stream = nm_stream_offline.Stream(
            sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
        )


def test_different_segment_lengths():
    segment_length_features_ms = 800

    arr_test = np.random.random([2, 1200])
    settings, nm_channels = get_example_settings(arr_test)

    settings["segment_length_features_ms"] = segment_length_features_ms
    settings["fft_settings"]["windowlength_ms"] = segment_length_features_ms

    stream = nm_stream_offline.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df_seglength_800 = stream.run(arr_test)

    segment_length_features_ms = 1000

    arr_test = np.random.random([2, 1200])
    settings, nm_channels = get_example_settings(arr_test)

    settings["segment_length_features_ms"] = segment_length_features_ms
    settings["fft_settings"]["windowlength_ms"] = segment_length_features_ms

    stream = nm_stream_offline.Stream(
        sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
    )

    df_seglength_1000 = stream.run(arr_test)
    # check the difference between time points

    assert (
        df_seglength_1000.iloc[0]["ch0-avgref_fft_theta_mean"]
        != df_seglength_800.iloc[0]["ch0-avgref_fft_theta_mean"]
    )
