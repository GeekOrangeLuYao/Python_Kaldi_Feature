from typing import Union

import numpy as np

from feature.feature_fbank import FbankComputer, FbankOptions
from feature.feature_mfcc import MfccComputer, MfccOptions
from feature.feature_window import FeatureWindowFunction, compute_num_frames

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
        cols_out = self.feature_computer.get_dim()

        if rows_out == 0:
            return np.array([])

