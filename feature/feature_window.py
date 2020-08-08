import numpy as np

from base.math_util import round_up_to_nearest_power_of_two


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
