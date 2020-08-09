from typing import Union

import numpy as np

from feature.feature_fbank import FbankComputer, FbankOptions
from feature.feature_mfcc import MfccComputer, MfccOptions
from feature.feature_window import FeatureWindowFunction, compute_num_frames, extract_window

FeatureComputer = Union[FbankComputer, MfccComputer]
FeatureOptions = Union[FbankOptions, MfccOptions]


class FeatureExtractor(object):

    def __init__(self,
                 opts: FeatureOptions,
                 feature_computer: FeatureComputer):
        self.feature_computer = feature_computer
        self.window_function = FeatureWindowFunction(opts.get_frame_options())

    def compute_features(self,
                         wave: np.ndarray,
                         sample_freq, ) -> np.ndarray:
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
