import enum
import numpy as np
from typing import Iterable
from scipy import signal, ndimage

from py_neuromodulation import nm_features_abc, nm_filter


class Burst(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.threshold = self.s["burst_settings"]["threshold"]
        self.time_duration_s = self.s["burst_settings"]["time_duration_s"]
        self.fband_names = self.s["burst_settings"]["frequency_bands"]
        self.f_ranges = [
            self.s["frequency_ranges_hz"][fband_name]
            for fband_name in self.fband_names
        ]
        self.seglengths = np.floor(
            self.sfreq
            / 1000
            * np.array(
                [
                    self.s["bandpass_filter_settings"]["segment_lengths_ms"][
                        fband
                    ]
                    for fband in self.fband_names
                ]
            )
        ).astype(int)

        self.num_max_samples_ring_buffer = int(
            self.sfreq * self.time_duration_s
        )

        self.bandpass_filter = nm_filter.BandPassFilter(
            f_ranges=self.f_ranges,
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

        # Create circular buffer array for previous time_duration_s
        self.data_buffer = np.empty((len(self.ch_names), len(self.fband_names), 0), dtype=np.float64)

    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):
        assert isinstance(
            settings["burst_settings"]["threshold"], (float, int)
        ), f"burst settings threshold needs to be type int or float, got: {settings['burst_settings']['threshold']}"
        assert (
            0 < settings["burst_settings"]["threshold"] < 100
        ), f"burst setting threshold needs to be between 0 and 100, got: {settings['burst_settings']['threshold']}"
        assert isinstance(
            settings["burst_settings"]["time_duration_s"], (float, int)
        ), f"burst settings time_duration_s needs to be type int or float, got: {settings['burst_settings']['time_duration_s']}"
        assert (
            settings["burst_settings"]["time_duration_s"] > 0
        ), f"burst setting time_duration_s needs to be greater than 0, got: {settings['burst_settings']['time_duration_s']}"

        for fband_burst in settings["burst_settings"]["frequency_bands"]:
            assert fband_burst in list(
                settings["frequency_ranges_hz"].keys()
            ), f"bursting {fband_burst} needs to be defined in settings['frequency_ranges_hz']"

        for burst_feature in settings["burst_settings"][
            "burst_features"
        ].keys():
            assert isinstance(
                settings["burst_settings"]["burst_features"][burst_feature],
                bool,
            ), (
                f"bursting feature {burst_feature} needs to be type bool, "
                f"got: {settings['burst_settings']['burst_features'][burst_feature]}"
            )

    def calc_feature(self, data: np.array, features_compute: dict) -> dict:
        # filter_data returns (n_channels, n_fbands, n_samples)
        filtered_data = np.abs(signal.hilbert(self.bandpass_filter.filter_data(data), axis=2))
        n_channels, n_fbands, n_samples = filtered_data.shape

        # Update buffer array
        excess = max(0, self.data_buffer.shape[2] + n_samples - self.num_max_samples_ring_buffer)
        self.data_buffer = np.concatenate((self.data_buffer[:,:,excess:], filtered_data), axis=2)

        # Detect bursts as values above threshold
        burst_thr = np.percentile(self.data_buffer, q=self.threshold, axis=2, keepdims=True)
        bursts = filtered_data >= burst_thr
        
        """ Label each burst with a unique id"""
        # Add a zero between each data series, flatten and get unique label for each burst
        burst_labels, n_labels = ndimage.label(
            np.concatenate((bursts, np.zeros(bursts.shape[:2]+(1,), dtype=bool)), 
                           axis=2)
            .flatten()
        )
        # Go back to original shape and remove zeros, and get mean for each burst
        burst_labels = burst_labels.reshape(n_channels, n_fbands, n_samples+1)[:,:,:-1]
        
        # Get length of each burst, so we can get the max burst duration for each series
        burst_lengths = ndimage.sum_labels(bursts, burst_labels, index=range(1,n_labels+1))
        
        # The mean is actually a mean of means, so we need the mean for each individual burst
        burst_amplitude_mean = ndimage.mean(filtered_data, burst_labels, index=range(1,n_labels+1))

        bursts_cumsum = np.cumsum(bursts, axis = 2) # Use prefix sum to get burst lengths
        # Detect falling edges as places where the sum stops changing
        falling_edges = np.concatenate(
            (np.zeros((n_channels, n_fbands, 2), dtype=bool),
             np.diff(bursts_cumsum, n=2) < 0), 
             axis = 2)

        # ! Do I need 
        # num_bursts = np.sum(np.diff(bursts, axis = 2), axis = 2) // 2
        num_bursts = np.sum(falling_edges, axis = 2)

        burst_duration_mean = np.sum(bursts, axis = 2) / num_bursts

        burst_duration_max = np.empty((n_channels, n_fbands))
        
        burst_amplitude_max = (filtered_data * bursts).max(axis=2)
        
        burst_rate_per_s = burst_duration_mean/ self.s["segment_length_features_ms"] / 1000
        end_in_burst = bursts[:,:,-1]
            
        """ Create dictionary to return """
        for ch_i, ch in enumerate(self.ch_names):
            for fb_i, fb in enumerate(self.fband_names):
                
                num_bursts

                
                features_compute[f"{ch}_bursts_{fb}_duration_mean"] = burst_duration_mean[ch_i,fb_i]
                features_compute[f"{ch}_bursts_{fb}_duration_max"] = burst_duration_max[ch_i,fb_i]

                features_compute[f"{ch}_bursts_{fb}_amplitude_mean"] = burst_amplitude_mean[ch_i,fb_i]
                features_compute[f"{ch}_bursts_{fb}_amplitude_max"] = burst_amplitude_max[ch_i,fb_i]

                features_compute[f"{ch}_bursts_{fb}_burst_rate_per_s"] = burst_rate_per_s[ch_i,fb_i]

                features_compute[f"{ch}_bursts_{fb}_in_burst"] = end_in_burst[ch_i,fb_i]

        return features_compute

    @staticmethod
    def get_burst_amplitude_length(
        beta_averp_norm, burst_thr: float, sfreq: float
    ):
        """
        Analysing the duration of beta burst
        """
        bursts = np.zeros((beta_averp_norm.shape[0] + 1), dtype=bool)
        bursts[1:] = beta_averp_norm >= burst_thr
        deriv = np.diff(bursts)
        
        burst_length = []
        burst_amplitude = []

        burst_time_points = np.where(deriv==True)[0]

        for i in range(burst_time_points.size//2):
            burst_length.append(burst_time_points[2 * i + 1] - burst_time_points[2 * i])
            burst_amplitude.append(beta_averp_norm[burst_time_points[2 * i] : burst_time_points[2 * i + 1]])
            
        # the last burst length (in case isburst == True) is omitted,
        # since the true burst length cannot be estimated
        burst_length = np.array(burst_length) / sfreq

        return burst_amplitude, burst_length
    
