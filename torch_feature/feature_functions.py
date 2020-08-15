import numpy as np
import torch

from feature.feature_window import FeatureWindowFunction, FrameExtractionOptions


def init_preemph_matrix(frame_length_padded, preemph_coeff) -> np.ndarray:
    mat = np.eye(frame_length_padded) - preemph_coeff * np.eye(frame_length_padded, k=1)
    mat[0, 0] = 1 - preemph_coeff
    return mat


def init_window_function(opts: FrameExtractionOptions):
    feature_window_function = FeatureWindowFunction(opts)
    return feature_window_function.get_window()


