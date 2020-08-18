from typing import Union

import numpy as np

from feature.feature_fbank import FbankComputer, FbankOptions
from feature.feature_mfcc import MfccComputer, MfccOptions
from feature.feature_spectrogram import SpectrogramComputer, SpectrogramOptions
from feature.feature_window import FeatureWindowFunction, compute_num_frames, extract_window
from feature.feature_config import OptionsParser

FeatureComputer = Union[FbankComputer, MfccComputer, SpectrogramComputer]
FeatureOptions = Union[FbankOptions, MfccOptions, SpectrogramOptions]


def build_feature_computer(feature_type, opts: FeatureOptions) -> FeatureComputer:
    if feature_type == "mfcc":
        return MfccComputer(opts)
    elif feature_type == "fbank":
        return FbankComputer(opts)
    elif feature_type == "stft":
        return SpectrogramComputer(opts)
    else:
        raise ValueError(f"Wrong {feature_type}")


def build_feature_options(feature_type, option_parser: OptionsParser) -> FeatureOptions:
    if feature_type == "mfcc":
        options = MfccOptions()
    elif feature_type == "fbank":
        options = FbankOptions()
    elif feature_type == "stft":
        options = SpectrogramOptions()
    else:
        raise ValueError(f"Feature type {feature_type} do not exist")
    options.register(option_parser)
    return options


class FeatureExtractor(object):

    def __init__(self, feature_type, option_parser: OptionsParser):
        self.feature_type = feature_type
        self.feature_options = build_feature_options(feature_type, option_parser)
        self.feature_computer = build_feature_computer(feature_type, self.feature_options)
        self.window_function = FeatureWindowFunction(self.feature_options.get_frame_options())

    def compute_features(self,
                         wave: np.ndarray,
                         sample_freq) -> np.ndarray:
        # TODO: use the downsample
        assert sample_freq == self.feature_computer.get_frame_extraction_options().samp_freq
        return self.compute(wave)

    def compute(self, wave: np.ndarray) -> np.ndarray:
        rows_out = compute_num_frames(wave.shape[0], self.feature_computer.get_frame_extraction_options())
        cols_out = self.feature_computer.dim()

        if rows_out == 0:
            return np.array([])

        output = np.zeros((rows_out, cols_out))
        use_raw_log_energy = self.feature_computer.need_raw_log_energy()
        for r in range(rows_out):
            raw_log_energy = 0.0
            if use_raw_log_energy:
                window, raw_log_energy = extract_window(0, wave, r,
                                                        self.feature_computer.get_frame_extraction_options(),
                                                        self.window_function, raw_log_energy)
            else:
                window, raw_log_energy = extract_window(0, wave, r,
                                                        self.feature_computer.get_frame_extraction_options(),
                                                        self.window_function, None)
                raw_log_energy = 0.0

            output[r, :] = self.feature_computer.compute(raw_log_energy, window)
        return output
