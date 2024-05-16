from collections.abc import Iterable
import numpy as np

from pydantic import BaseModel, field_validator
from typing import TYPE_CHECKING

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_types import FeatureSelector

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings
    from py_neuromodulation.nm_kalmanfilter import KalmanSettings

class OscillatoryFeatures(FeatureSelector):
    mean: bool = True
    median: bool = False
    std: bool = False
    max: bool = False


class OscillatorySettings(BaseModel):
    windowlength_ms: int
    log_transform: bool
    features: OscillatoryFeatures = OscillatoryFeatures(
        mean=True, median=False, std=False, max=False
    )
    return_spectrum: bool = False


class BandpowerFeatures(FeatureSelector):
    activity: bool = True
    mobility: bool = False
    complexity: bool = False


class BandpassSettings(BaseModel):
    segment_lengths_ms: dict[str, int] = {
        "theta": 1000,
        "alpha": 500,
        "low beta": 333,
        "high beta": 333,
        "low gamma": 100,
        "high gamma": 100,
        "HFA": 100,
    }
    bandpower_features: BandpowerFeatures
    log_transform: bool = True
    kalman_filter: bool = False

    @field_validator("bandpower_features")
    @classmethod
    def bandpower_features_validator(cls, bandpower_features: BandpowerFeatures):
        assert (
            len(bandpower_features.get_enabled()) > 0
        ), "Set at least one bandpower_feature to True."
        return bandpower_features


class OscillatoryFeature(NMFeature):
    def __init__(
        self, settings: 'NMSettings', ch_names: Iterable[str], sfreq: float
    ) -> None:
        # self.settings = settings
        self.settings: OscillatorySettings  # Assignment in subclass __init__
        self.osc_feature_name: str  # Required for output

        self.sfreq = sfreq
        self.ch_names = ch_names
        self.KF_dict: dict = {}

        self.frequency_ranges = settings.frequency_ranges_hz

        # TONI: This could be tested at the NMSettings level, or is it only needed when oscillatory features are enabled?
        assert (
            fb[0] < sfreq / 2 and fb[1] < sfreq / 2
            for fb in settings.frequency_ranges_hz.values()
        ), (
            "the frequency band ranges need to be smaller than the nyquist frequency"
            f"got sfreq = {sfreq} and fband ranges {settings.frequency_ranges_hz}"
        )

        assert self.settings.windowlength_ms <= settings.segment_length_features_ms, (
            f"oscillatory feature windowlength_ms = ({self.settings.windowlength_ms})"
            f"needs to be smaller than"
            f"settings['segment_length_features_ms'] = {settings.segment_length_features_ms}",
        )

    def estimate_osc_features(
        self,
        features_compute: dict,
        data: np.ndarray,
    ):
        for feature_est_name in self.settings.features.get_enabled():
            col_name = f"{self.osc_feature_name}_{feature_est_name}"
            match feature_est_name:
                case "mean":
                    features_compute[col_name] = np.nanmean(data)
                case "median":
                    features_compute[col_name] = np.nanmedian(data)
                case "std":
                    features_compute[col_name] = np.nanstd(data)
                case "max":
                    features_compute[col_name] = np.nanmax(data)

        return features_compute


class FFT(OscillatoryFeature):
    def __init__(
        self,
        settings: 'NMSettings',
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        from scipy.fft import rfftfreq

        self.osc_feature_name = "fft"
        self.settings = settings.fft_settings
        # super.__init__ needs osc_feature_name and settings
        super().__init__(settings, ch_names, sfreq)

        window_ms = self.settings.windowlength_ms

        self.window_samples = int(-np.floor(window_ms / 1000 * sfreq))
        self.freqs = rfftfreq(-self.window_samples, 1 / np.floor(self.sfreq))

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.frequency_ranges.items():
                idx_range = np.where(
                    (self.freqs >= f_range[0]) & (self.freqs < f_range[1])
                )[0]
                self.feature_params.append((ch_idx, idx_range))

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        data = data[:, self.window_samples :]

        from scipy.fft import rfft

        Z = np.abs(rfft(data))

        if self.settings.log_transform:
            Z = np.log10(Z)

        for ch_idx, idx_range in self.feature_params:
            Z_ch = Z[ch_idx, idx_range]

            features_compute = self.estimate_osc_features(
                features_compute,
                Z_ch,
            )

        for ch_idx, ch_name in enumerate(self.ch_names):
            if self.settings.return_spectrum:
                features_compute.update(
                    {
                        f"{ch_name}_fft_psd_{str(f)}": Z[ch_idx][idx]
                        for idx, f in enumerate(self.freqs.astype(int))
                    }
                )

        return features_compute


class Welch(OscillatoryFeature):
    def __init__(
        self,
        settings: 'NMSettings',
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        self.osc_feature_name = "welch"
        self.settings = settings.welch_settings
        # super.__init__ needs osc_feature_name and settings
        super().__init__(settings, ch_names, sfreq)

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.frequency_ranges.items():
                self.feature_params.append((ch_idx, f_range))

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        from scipy.signal import welch

        freqs, Z = welch(
            data,
            fs=self.sfreq,
            window="hann",
            nperseg=self.sfreq,
            noverlap=None,
        )

        if self.settings.log_transform:
            Z = np.log10(Z)

        for ch_idx, f_range in self.feature_params:
            Z_ch = Z[ch_idx]

            idx_range = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[0]

            features_compute = self.estimate_osc_features(
                features_compute,
                Z_ch[idx_range],
            )

        for ch_idx, ch_name in enumerate(self.ch_names):
            if self.settings.return_spectrum:
                features_compute.update(
                    {
                        f"{ch_name}_welch_psd_{str(f)}": Z[ch_idx][idx]
                        for idx, f in enumerate(freqs.astype(int))
                    }
                )

        return features_compute


class STFT(OscillatoryFeature):
    def __init__(
        self,
        settings: 'NMSettings',
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        self.osc_feature_name = "stft"
        self.settings = settings.stft_settings
        # super.__init__ needs osc_feature_name and settings
        super().__init__(settings, ch_names, sfreq)

        self.nperseg = self.settings.windowlength_ms

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.frequency_ranges.items():
                self.feature_params.append((ch_idx, f_range))

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        from scipy.signal import stft

        freqs, _, Zxx = stft(
            data,
            fs=self.sfreq,
            window="hamming",
            nperseg=self.nperseg,
            boundary="even",
        )
        Z = np.abs(Zxx)
        if self.settings.log_transform:
            Z = np.log10(Z)
        for ch_idx, feature_name, f_range in self.feature_params:
            Z_ch = Z[ch_idx]
            idx_range = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[0]

            features_compute = self.estimate_osc_features(
                features_compute,
                Z_ch[idx_range, :],
            )

        for ch_idx, ch_name in enumerate(self.ch_names):
            if self.settings.return_spectrum:
                Z_ch_mean = Z[ch_idx].mean(axis=1)
                features_compute.update(
                    {
                        f"{ch_name}_stft_psd_{str(f)}": Z_ch_mean[idx]
                        for idx, f in enumerate(freqs.astype(int))
                    }
                )

        return features_compute


class BandPower(NMFeature):
    def __init__(
        self,
        settings: 'NMSettings',
        ch_names: Iterable[str],
        sfreq: float,
        use_kf: bool | None = None,
    ) -> None:
        self.bp_settings: BandpassSettings = settings.bandpass_filter_settings
        self.kalman_filter_settings: KalmanSettings = settings.kalman_filter_settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.KF_dict: dict = {}

        from py_neuromodulation.nm_filter import MNEFilter

        self.bandpass_filter = MNEFilter(
            f_ranges=list(settings.frequency_ranges_hz.values()),
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

        if use_kf or (use_kf is None and self.bp_settings.kalman_filter):
            self.init_KF("bandpass_activity")

        seglengths = self.bp_settings.segment_lengths_ms

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for f_band_idx, f_band in enumerate(settings.frequency_ranges_hz.keys()):
                seglength_ms = seglengths[f_band]
                seglen = int(np.floor(self.sfreq / 1000 * seglength_ms))
                for bp_feature in self.bp_settings.bandpower_features.get_enabled():
                    feature_name = "_".join([ch_name, "bandpass", bp_feature, f_band])
                    self.feature_params.append(
                        (
                            ch_idx,
                            f_band_idx,
                            seglen,
                            bp_feature,
                            feature_name,
                        )
                    )

    def init_KF(self, feature: str) -> None:
        from py_neuromodulation.nm_kalmanfilter import define_KF

        for f_band in self.kalman_filter_settings.frequency_bands:
            for channel in self.ch_names:
                self.KF_dict["_".join([channel, feature, f_band])] = define_KF(
                    self.kalman_filter_settings.Tp,
                    self.kalman_filter_settings.sigma_w,
                    self.kalman_filter_settings.sigma_v,
                )

    def update_KF(self, feature_calc: np.floating, KF_name: str) -> np.floating:
        if KF_name in self.KF_dict:
            self.KF_dict[KF_name].predict()
            self.KF_dict[KF_name].update(feature_calc)
            feature_calc = self.KF_dict[KF_name].x[0]
        return feature_calc

    # @staticmethod
    # def test_settings(settings: dict, ch_names: Iterable[str], sfreq: float):

    # TODO needs to be checked at the NMSettings level or context needs to be passed
    # for fband_name, seg_length_fband in settings["bandpass_filter_settings"][
    #     "segment_lengths_ms"
    # ].items():

    #     assert seg_length_fband <= settings["segment_length_features_ms"], (
    #         f"segment length {seg_length_fband} needs to be smaller than "
    #         f" settings['segment_length_features_ms'] = {settings['segment_length_features_ms']}"
    #     )

    # for fband_name in list(settings["frequency_ranges_hz"].keys()):
    #     assert fband_name in list(
    #         settings["bandpass_filter_settings"]["segment_lengths_ms"].keys()
    #     ), (
    #         f"frequency range {fband_name} "
    #         "needs to be defined in settings['bandpass_filter_settings']['segment_lengths_ms']"
    #     )

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        data = self.bandpass_filter.filter_data(data)

        for (
            ch_idx,
            f_band_idx,
            seglen,
            bp_feature,
            feature_name,
        ) in self.feature_params:
            features_compute[feature_name] = self.calc_bp_feature(
                bp_feature, feature_name, data[ch_idx, f_band_idx, -seglen:]
            )

        return features_compute

    def calc_bp_feature(self, bp_feature, feature_name, data):
        match bp_feature:
            case "activity":
                feature_calc = np.var(data)
                if self.bp_settings.log_transform:
                    feature_calc = np.log10(feature_calc)
                if self.KF_dict:
                    feature_calc = self.update_KF(feature_calc, feature_name)
            case "mobility":
                feature_calc = np.sqrt(np.var(np.diff(data)) / np.var(data))
            case "complexity":
                feature_calc = self.calc_complexity(data)
            case _:
                raise ValueError(f"Unknown bandpower feature: {bp_feature}")

        return np.nan_to_num(feature_calc)

    @staticmethod
    def calc_complexity(data: np.ndarray) -> float:
        dat_deriv = np.diff(data)
        deriv_variance = np.var(dat_deriv)
        mobility = np.sqrt(deriv_variance / np.var(data))
        dat_deriv_2_var = np.var(np.diff(dat_deriv))
        deriv_mobility = np.sqrt(dat_deriv_2_var / deriv_variance)

        return deriv_mobility / mobility
