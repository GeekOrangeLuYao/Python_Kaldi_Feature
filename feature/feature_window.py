from typing import Tuple
import numpy as np

from base.math_util import round_up_to_nearest_power_of_two
from base.math_util import epsilon
from feature.feature_config import OptionsParser


class FrameExtractionOptions(object):

    def __init__(self,
                 samp_freq=16000,
                 frame_shift=10.0,
                 frame_length=25.0,
                 dither=1.0,
                 preemph_coeff=0.97,
                 remove_dc_offset=True,
                 window_type="povey",
                 blackman_coeff=0.42,
                 snip_edges=True,
                 allow_downsample=False):
        self.samp_freq = samp_freq
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.dither = dither
        self.preemph_coeff = preemph_coeff
        self.remove_dc_offset = remove_dc_offset
        self.window_type = window_type
        self.blackman_coeff = blackman_coeff
        self.snip_edges = snip_edges
        self.allow_downsample = allow_downsample

    def register(self, option_parser: OptionsParser):
        self.samp_freq = option_parser.get("samp_freq", 16000, type_function=np.int)
        self.frame_shift = option_parser.get("frame_shift", 10.0, type_function=np.float)
        self.frame_length = option_parser.get("frame_length", 25.0, type_function=np.float)
        self.dither = option_parser.get("dither", 1.0, type_function=np.float)
        self.preemph_coeff = option_parser.get("preemph_coeff", 0.97, type_function=np.float)
        self.remove_dc_offset = option_parser.get("remove_dc_offset", True, type_function=np.bool)
        self.window_type = option_parser.get("window_type", "povey", type_function=np.str)
        self.blackman_coeff = option_parser.get("blackman_coeff", 0.42, type_function=np.float)
        self.snip_edges = option_parser.get("snip_edges", True, type_function=np.bool)
        self.allow_downsample = option_parser.get("allow_downsample", False, type_function=np.bool)

    def get_win_shift(self):
        return np.int(self.samp_freq * 0.001 * self.frame_shift)

    def get_win_size(self):
        return np.int(self.samp_freq * 0.001 * self.frame_length)

    def get_padded_window_size(self):
        """
            padded the window size to 2^k, please check in the round_up_to_nearest_power_of_two function
            Two make the fft effectively, always use this methods
        """
        return round_up_to_nearest_power_of_two(self.get_win_size())


class FeatureWindowFunction(object):

    def __init__(self, opts: FrameExtractionOptions):
        self.opts = opts
        self.window = None
        self._init()

    def _init(self):
        frame_length = self.opts.get_win_size()
        assert frame_length > 0

        self.window = np.zeros((frame_length,))
        a = 2 * np.pi / (frame_length - 1)

        for i in range(frame_length):
            i_fl = float(i)
            if self.opts.window_type == "hanning":
                self.window[i] = 0.5 - 0.5 * np.cos(a * i_fl)
            elif self.opts.window_type == "hamming":
                self.window[i] = 0.54 - 0.46 * np.cos(a * i_fl)
            elif self.opts.window_type == "povey":
                self.window[i] = np.power(0.5 - 0.5 * np.cos(a * i_fl), 0.85)
            elif self.opts.window_type == "rectangular":
                self.window[i] = 1.
            elif self.opts.window_type == "blackman":
                self.window[i] = self.opts.blackman_coeff - 0.5 * np.cos(a * i_fl) + (
                        0.5 - self.opts.blackman_coeff) * np.cos(2 * a * i_fl)
            else:
                raise ValueError(f"Unknown window_type = {self.opts.window_type}")

    def get_window(self):
        return self.window


def first_sample_of_frame(frame, opts: FrameExtractionOptions):
    frame_shift = opts.get_win_shift()
    if opts.snip_edges:
        return frame * frame_shift
    else:
        midpoint_of_frame = frame_shift * frame + frame_shift // 2
        beginning_of_frame = midpoint_of_frame - opts.get_win_size() // 2
        return beginning_of_frame


def compute_num_frames(num_samples, opts: FrameExtractionOptions):
    frame_shift = opts.get_win_shift()
    frame_length = opts.get_win_size()

    if opts.snip_edges:
        if num_samples < frame_length:
            return 0
        else:
            return 1 + ((num_samples - frame_length) // frame_shift)
    else:
        num_frames = (num_samples + (frame_shift // 2)) // frame_shift
        end_sample_of_last_frame = first_sample_of_frame(num_frames - 1, opts) + frame_length

        while num_frames > 0 and end_sample_of_last_frame > num_samples:
            # TODO: Kaldi optimized more for clarity than efficiency, try to change it
            num_frames -= 1
            end_sample_of_last_frame -= frame_shift
        return num_frames


def dither(wave_form, dither_value):
    if dither_value == 0.0:
        return
    dim = wave_form.shape[0]
    wave_form += dither_value * np.random.rand(dim)

    # TODO: check this function, especially the random function
    return wave_form


def preemphasize(wave_form: np.ndarray, preemph_coeff):
    if preemph_coeff == 0.0:
        return None
    assert 0.0 <= preemph_coeff <= 1.0
    i = wave_form.shape[0] - 1
    while i > 0:
        wave_form[i] -= preemph_coeff * wave_form[i - 1]
        i -= 1
    wave_form[0] -= preemph_coeff * wave_form[0]
    return wave_form


def process_window(opts: FrameExtractionOptions,
                   window_function: FeatureWindowFunction,
                   window: np.ndarray,
                   log_energy_pre_window):
    # dither -> remove_dc_offset -> preemph
    frame_length = opts.get_win_size()

    # TODO: use dither
    # if opts.dither != 0.0:

    if opts.remove_dc_offset:
        window -= np.sum(window) / frame_length

    if log_energy_pre_window is not None:
        energy = np.max(np.dot(window, window), epsilon())
        log_energy_pre_window = np.log(energy)

    if opts.preemph_coeff != 0.0:
        window = preemphasize(window, opts.preemph_coeff)

    window = window * window_function.window
    return window, log_energy_pre_window


def extract_window(sample_offset,
                   wave: np.ndarray,
                   f,
                   opts: FrameExtractionOptions,
                   window_function: FeatureWindowFunction,
                   log_energy_pre_window):  # return window
    assert sample_offset >= 0 and wave.shape[0] != 0

    frame_length = opts.get_win_size()
    frame_length_padded = opts.get_padded_window_size()
    num_samples = sample_offset + wave.shape[0]
    start_sample = first_sample_of_frame(f, opts)
    end_sample = start_sample + frame_length

    if opts.snip_edges:
        assert start_sample >= sample_offset and end_sample <= num_samples
    else:
        assert sample_offset == 0 or start_sample >= sample_offset

    wave_start = np.int(start_sample - sample_offset)
    wave_end = wave_start + frame_length

    window = np.zeros((frame_length_padded,))
    if wave_start >= 0 and wave_end <= wave.shape[0]:
        window[:frame_length] = np.copy(wave[wave_start: wave_end])
    else:
        wave_dim = wave.shape[0]
        for s in range(frame_length):
            s_in_wave = s + wave_start

            while s_in_wave < 0 or s_in_wave >= wave_dim:
                if s_in_wave < 0:
                    s_in_wave = - s_in_wave - 1
                else:
                    s_in_wave = 2 * wave_dim - 1 - s_in_wave
            window[s] = wave[s_in_wave]

    if frame_length_padded > frame_length:
        window[frame_length: frame_length_padded] = 0.0

    window[:frame_length], log_energy_pre_window = process_window(opts,
                                                                  window_function,
                                                                  window[:frame_length],
                                                                  log_energy_pre_window)

    return window, log_energy_pre_window
