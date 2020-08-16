import numpy as np
import torch

from feature.feature_window import FeatureWindowFunction, FrameExtractionOptions
from feature.mel_computations import MelBanksOptions, MelBanks


def init_preemph_matrix(frame_length_padded, preemph_coeff) -> np.ndarray:
    mat = np.eye(frame_length_padded) - preemph_coeff * np.eye(frame_length_padded, k=1)
    mat[0, 0] = 1 - preemph_coeff
    return mat


def init_window_function(opts: FrameExtractionOptions):
    feature_window_function = FeatureWindowFunction(opts)
    return feature_window_function.get_window()


def init_mel_banks(opts: MelBanksOptions = None, frame_opts: FrameExtractionOptions = None) -> np.ndarray:
    mel_banks = MelBanks(opts, frame_opts)
    return mel_banks.get_bins_matrix()
