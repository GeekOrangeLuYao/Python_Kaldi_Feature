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

    def __init__(self, order=2, window=2):
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

            cur_scales = np.zeros((self.scales[i].shape[0] + 2 * window,))
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


class SlidingWindowCmnOptions(object):
    def __init__(self,
                 cmn_window=600,
                 min_window=100,
                 max_warnings=5,
                 normalize_variance=False,
                 center=False):
        self.cmn_window = cmn_window
        self.min_window = min_window
        self.max_warnings = max_warnings
        self.normalize_variance = normalize_variance
        self.center = center

    def register(self, option_parser: OptionsParser):
        # self.order = option_parser.get("delta_order", 2, type_function=np.int)
        # self.window = option_parser.get("delta_window", 2, type_function=np.int)
        self.cmn_window = option_parser.get("cmn_window", 600, type_function=np.int)
        self.min_window = option_parser.get("min_window", 100, type_function=np.int)
        self.max_warnings = option_parser.get("max_warnings", 5, type_function=np.int)
        self.normalize_variance = option_parser.get("normalize_variance", "False", type_function=np.bool)
        self.center = option_parser.get("center", "False", type_function=np.bool)

    def check(self):
        assert self.cmn_window > 0
        if self.center:
            assert 0 < self.min_window <= self.cmn_window


def sliding_window_cmn(opts: SlidingWindowCmnOptions,
                       input: np.ndarray) -> np.ndarray:
    # remove `SlidingWindowCmnInternal` functions and just use `SlidingWindowCmn`
    opts.check()
    num_frames, dim = input.shape
    last_window_start = -1
    last_window_end = -1
    output = np.zeros_like(input)

    cur_sum = np.zeros((dim, ))
    cur_sumsq = np.zeros((dim, ))
    for t in range(num_frames):
        if opts.center:
            window_start = t - opts.cmn_window // 2
            window_end = window_start + opts.cmn_window
        else:
            window_start = t - opts.cmn_window
            window_end = t + 1

        if window_start < 0:
            window_end -= window_start
            window_start = 0

        if not opts.center:
            if window_end > t:
                window_end = max(t + 1, opts.min_window)

        if window_end > num_frames:
            window_start -= (window_end - num_frames)
            window_end = num_frames
            if window_start < 0:
                window_start = 0

        if last_window_start == -1:
            input_part = input[window_start: window_end, :]
            cur_sum += np.sum(input_part, axis=0)
            if opts.normalize_variance:
                cur_sumsq += np.diag(np.dot(input_part.T, input_part))
        else:
            if window_start > last_window_start:
                assert window_start == last_window_start + 1
                frame_to_remove = input[last_window_start, :]
                cur_sum -= frame_to_remove
                if opts.normalize_variance:
                    cur_sumsq -= frame_to_remove ** 2
            if window_end > last_window_end:
                assert window_end == last_window_end + 1
                frame_to_add = input[last_window_end, :]
                cur_sum += frame_to_add
                if opts.normalize_variance:
                    cur_sumsq += frame_to_add ** 2

        window_frames = window_end - window_start
        last_window_start = window_start
        last_window_end = window_end

        assert window_frames > 0
        output[t, :] = input[t, :] - 1. / window_frames * cur_sum

        if opts.normalize_variance:
            if window_frames == 1:
                output[t, :] = 0.0
            else:
                variance = np.copy(cur_sumsq)
                variance *= 1. / window_frames
                variance -= (window_frames * window_frames) * (cur_sum ** 2)
                # remove num_floored as usual
                variance = np.maximum(variance, 1e-10)
                variance = np.power(variance, -0.5)
                output[t, :] = output[t, :] * variance

    return output
