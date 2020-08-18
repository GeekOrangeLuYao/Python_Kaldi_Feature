import numpy as np

from feature.feature_window import FrameExtractionOptions
from feature.feature_config import OptionsParser
from base.math_util import epsilon, numeric_limits_float_min


class SpectrogramOptions(object):
    def __init__(self,
                 opts: FrameExtractionOptions = None,
                 energy_floor=0.0,
                 raw_energy=True):
        self.frame_opts = opts if opts is not None else FrameExtractionOptions()
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy

    def get_frame_options(self):
        return self.frame_opts

    def register(self, option_parser: OptionsParser):
        self.frame_opts.register(option_parser)
        self.energy_floor = option_parser.get("energy_floor", 0.0, np.float)
        self.raw_energy = option_parser.get("raw_energy", "True", np.bool)


class SpectrogramComputer(object):
    def __init__(self,
                 opts: SpectrogramOptions):
        self.opts = opts
        self.log_energy_floor = 0.0
        if opts.energy_floor > 0.0:
            self.log_energy_floor = np.log(opts.energy_floor)

    def compute(self,
                signal_log_energy,
                signal_frame: np.ndarray) -> np.ndarray:
        if not self.opts.raw_energy:
            signal_log_energy = np.log(np.log(max(np.dot(signal_frame, signal_frame), numeric_limits_float_min())))
        signal_frame = np.fft.fft(signal_frame)
        signal_frame = np.abs(signal_frame) ** 2
        signal_frame_dim = signal_frame.shape[0]

        power_spectrum = signal_frame[:signal_frame_dim // 2 + 1]
        power_spectrum = np.maximum(power_spectrum, epsilon())
        feature = np.log(power_spectrum)

        if self.opts.energy_floor > 0.0 and signal_log_energy < self.log_energy_floor:
            signal_log_energy = self.log_energy_floor
        feature[0] = signal_log_energy
        return feature

    def dim(self):
        return self.opts.get_frame_options().get_padded_window_size() // 2 + 1

    def need_raw_log_energy(self):
        return self.opts.raw_energy
