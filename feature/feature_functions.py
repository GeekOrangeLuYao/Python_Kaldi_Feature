import numpy as np

from feature.feature_config import OptionsParser

def compute_power_spectrum(waveform: np.ndarray) -> np.ndarray:
    """
        The kaldi srfft is very different, which is not suitable for numpy
        Abandon this function
    """
    dim = waveform.shape[0]

    half_dim = dim // 2
    first_energy = waveform[0] ** 2
    last_energy = waveform[1] ** 2

    for i in range(1, half_dim):
        # TODO: check the i << 1 and i << 1 | 1
        real = waveform[i << 1]
        imag = waveform[i << 1 | 1]
        waveform[i] = real ** 2 + imag ** 2

    waveform[0] = first_energy
    waveform[half_dim] = last_energy

    return waveform



class DeltaFeatureOptions(object):

    def __init__(self, order = 2, window = 2):
        self.order = order
        self.window = window

    def register(self, option_parser: OptionsParser):
        self.order = option_parser.get("delta_order", 2, type_function=np.int)
        self.window = option_parser.get("delta_window", 2, type_function=np.int)


class DeltaFeature(object):
    def __init__(self, opts: DeltaFeatureOptions) -> None:
        assert 0 <= opts.order < 1000
        assert 0 < opts.window < 1000
        self.opts = opts

        self.scales = [np.array([1.0])]

        for i in range(opts.order):
            window = opts.window
            prev_offset = (self.scales[i].shape[0] - 1) // 2
            cur_offset = prev_offset + window

            cur_scales = np.zeros((self.scales[i].shape[0] + 2 * window, ))
            normalizer = 0.0
            for j in range(-window, window + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    cur_scales[j + k + cur_offset] += float(j) * self.scales[k + prev_offset]
            cur_scales /= normalizer
            self.scales.append(cur_scales)
    
    def process(self, input_feats: np.ndarray, frame: int) -> np.ndarray:
        num_frames = input_feats.shape[0]
        feat_dim = input_feats.shape[1]

        output_feats = np.zeros((feat_dim * (self.opts.order + 1)))
        for i in range(self.opts.order + 1):
            max_offset = (self.scales[i] - 1) // 2
            for j in range(-max_offset, max_offset + 1):
                offset_frame = min(max(0, frame + j), num_frames - 1)
                scale = self.scales[i][j + max_offset]
                if scale != 0.0:
                    raise NotImplementedError
        # TODO