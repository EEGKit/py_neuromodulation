"""Test the nm_filter module."""
import numpy as np
from py_neuromodulation import nm_filter
import pytest


class TestBandPassFilterData:
    """Test filter_data method of BandPassFilter class."""

    @pytest.mark.parametrize(
        "filter_length",
        ["999ms", "1999ms", "3999ms", "2s"],
    )
    def test_filter_length(self, filter_length) -> None:
        """Test different filter lengths."""
        f_ranges = [
            [4, 8],
        ]
        sfreq = 4000
        duration = 10
        times = np.linspace(0, duration, int(duration * sfreq))
        bandpass_filter = nm_filter.BandPassFilter(
            f_ranges=f_ranges,
            sfreq=sfreq,
            filter_length=filter_length,
            l_trans_bandwidth=4,
            h_trans_bandwidth=4,
            verbose=None,
        )
        oscill_freqs = 50
        data = np.sin(2 * np.pi * times * oscill_freqs)
        data_filtered = bandpass_filter.filter_data(data)
        assert data_filtered.shape == (
            1,
            len(f_ranges),
            duration * sfreq,
        )

    def test_filter_1d(self) -> None:
        """Test filtering of 1d array with multiple frequency ranges."""
        f_ranges = [
            [4, 8],
            [8, 12],
            [13, 35],
            [60, 200],
            [200, 500],
        ]
        sfreq = 4000
        duration = 10
        times = np.linspace(0, duration, int(duration * sfreq))
        bandpass_filter = nm_filter.BandPassFilter(
            f_ranges=f_ranges,
            sfreq=sfreq,
            filter_length="999ms",
            l_trans_bandwidth=4,
            h_trans_bandwidth=4,
            verbose=None,
        )
        oscill_freqs = 50
        data = np.sin(2 * np.pi * times * oscill_freqs)
        data_filtered = bandpass_filter.filter_data(data)
        assert data_filtered.shape == (
            1,
            len(f_ranges),
            duration * sfreq,
        )

    def test_filter_2d(self) -> None:
        """Test filtering of 2d array with multiple frequency ranges."""
        f_ranges = [
            [4, 8],
            [8, 12],
            [13, 35],
            [60, 200],
            [200, 500],
        ]
        sfreq = 4000
        duration = 10
        times = np.linspace(0, duration, int(duration * sfreq))
        bandpass_filter = nm_filter.BandPassFilter(
            f_ranges=f_ranges,
            sfreq=sfreq,
            filter_length="999ms",
            l_trans_bandwidth=4,
            h_trans_bandwidth=4,
            verbose=None,
        )
        oscill_freqs = np.expand_dims(np.arange(10, 51, 10), axis=-1)
        data = np.sin(2 * np.pi * times * oscill_freqs)
        data_filtered = bandpass_filter.filter_data(data)
        assert data_filtered.shape == (
            oscill_freqs.shape[0],
            len(f_ranges),
            duration * sfreq,
        )


if __name__ == "__main__":
    test = TestBandPassFilterData()
    # test.test_filter_length()
    # test.test_filter_1d()
    # test.test_filter_2d()
