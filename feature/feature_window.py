from base.math_util import RoundUpToNearestPowerOfTwo

class FrameExtractionOptions(object):

    def __init__(self,
                 samp_freq = 16000,
                 frame_shift_ms = 10.0,
                 frame_length_ms = 25.0,
                 dither = 1.0,
                 preemph_coeff = 0.97,
                 remove_dc_offset = True,
                 window_type = "povey",
                 round_to_power_of_two = True,
                 blackman_coeff = 0.42,
                 snip_edges = True,
                 allow_downsample = False) -> None:
        self.samp_freq = samp_freq
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.dither = dither
        self.preemph_coeff = preemph_coeff
        self.remove_dc_offset = remove_dc_offset
        self.window_type = window_type
        self.round_to_power_of_two = round_to_power_of_two
        self.blackman_coeff = blackman_coeff
        self.snip_edges = snip_edges
        self.allow_downsample = allow_downsample
    
    def get_window_shift(self) -> int:
        return int(self.samp_freq * 0.001 * self.frame_shift_ms)
    
    def get_window_size(self) -> int:
        return int(self.samp_freq * 0.001 * self.frame_length_ms)
    
    def get_padded_window_size(self) -> int:
        if self.round_to_power_of_two:
            return RoundUpToNearestPowerOfTwo(self.get_window_size())
        else:
            return self.get_padded_window_size()