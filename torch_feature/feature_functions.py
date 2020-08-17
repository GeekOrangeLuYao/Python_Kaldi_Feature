import numpy as np
import torch

from feature.feature_window import FeatureWindowFunction, FrameExtractionOptions
from feature.feature_window import compute_num_frames, first_sample_of_frame
from feature.mel_computations import MelBanksOptions, MelBanks
from feature.feature_mfcc import MfccOptions
from matrix.matrix_functions import compute_dct_matrix


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


def init_dct_matrix(opts:MfccOptions):
    num_bins = opts.mel_opts.num_bins
    dct_matrix = np.zeros((num_bins, num_bins))
    dct_matrix = compute_dct_matrix(dct_matrix)
    return dct_matrix[:opts.num_ceps, :num_bins]


def wave_reflection_padding(wave_len, opts: FrameExtractionOptions) -> np.ndarray:
    frame_length = opts.get_win_size()
    num_frames = compute_num_frames(wave_len, opts)
    begin = first_sample_of_frame(0, opts)
    end = first_sample_of_frame(num_frames - 1, opts) + frame_length
    assert begin < 0 and end >= wave_len, f"win_len {opts.get_win_size()} and win_shift {opts.get_win_shift()} are not suitable"

    reflection_padding_matrix = np.zeros((wave_len, end - begin))
    reflection_padding_matrix[:-begin, :-begin] = np.flipud(np.eye(-begin))
    reflection_padding_matrix[:, -begin:-begin + wave_len] = np.eye(wave_len)
    reflection_padding_matrix[2 * wave_len - end:, -begin + wave_len:] = np.flipud(np.eye(end - wave_len))

    return reflection_padding_matrix