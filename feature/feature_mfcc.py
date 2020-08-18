"""
    Compute Mfcc Feature

"""

import numpy as np

from feature.feature_window import FrameExtractionOptions
from feature.mel_computations import MelBanks, MelBanksOptions, compute_lifter_coeffs
from feature.feature_config import OptionsParser
from base.math_util import epsilon, numeric_limits_float_min
from matrix.matrix_functions import compute_dct_matrix


class MfccOptions(object):
    def __init__(self,
                 frame_opts: FrameExtractionOptions = None,
                 mel_opts: MelBanksOptions = MelBanksOptions(num_bins=23)) -> None:
        self.frame_opts = frame_opts if frame_opts is not None else FrameExtractionOptions()
        self.mel_opts = mel_opts if mel_opts is not None else MelBanksOptions()

        self.num_ceps = 13
        self.use_energy = True
        self.energy_floor = 0.0
        self.raw_energy = True
        self.cepstral_lifter = 22.0

    def register(self, option_parser: OptionsParser):
        self.frame_opts.register(option_parser)
        self.mel_opts.register(option_parser)

        self.num_ceps = option_parser.get("num_ceps", 13, type_function=np.int)
        self.use_energy = option_parser.get("use_energy", "True", type_function=np.bool)
        self.energy_floor = option_parser.get("energy_floor", 0.0, type_function=np.float)  # 0.0
        self.raw_energy = option_parser.get("raw_energy", "True", type_function=np.bool)
        self.cepstral_lifter = option_parser.get("cepstral_lifter", 22.0, type_function=np.float)

    def get_frame_options(self):
        return self.frame_opts


class MfccComputer(object):

    def __init__(self,
                 opts: MfccOptions) -> None:
        self.opts = opts
        self.mel_energies = np.zeros((opts.mel_opts.num_bins,))

        num_bins = opts.mel_opts.num_bins
        if opts.num_ceps > num_bins:
            raise ValueError(f"num-ceps {opts.num_ceps} cannot be larger than num-mel-bins {num_bins}.")

        dct_matrix = np.zeros((num_bins, num_bins))
        dct_matrix = compute_dct_matrix(dct_matrix)
        self.dct_matrix = dct_matrix[:opts.num_ceps, :num_bins]

        self.lifter_coeffs = None
        if opts.cepstral_lifter != 0.0:
            self.lifter_coeffs = np.zeros((opts.num_ceps,))
            self.lifter_coeffs = compute_lifter_coeffs(opts.cepstral_lifter, self.lifter_coeffs)

        if opts.energy_floor > 0.0:
            self.log_energy_floor = np.log(opts.energy_floor)

        self.mel_banks = None
        self.get_mel_banks()

    def compute(self,
                signal_log_energy,
                signal_frame: np.ndarray) -> np.ndarray:
        assert signal_frame.shape[0] == self.opts.frame_opts.get_padded_window_size()

        mel_banks: MelBanks = self.get_mel_banks()

        if self.opts.use_energy and not self.opts.raw_energy:
            signal_log_energy = np.log(max(np.dot(signal_frame, signal_frame), numeric_limits_float_min()))

        # Do srfft, and ComputePowerSpectrum
        signal_frame = np.fft.fft(signal_frame)
        # print(f"signal_frame:\n{signal_frame}")
        signal_frame = np.abs(signal_frame) ** 2  # get energy

        signal_frame_dim = signal_frame.shape[0]
        power_spectrum = signal_frame[:signal_frame_dim // 2 + 1]

        mel_energies = mel_banks.compute(power_spectrum)
        mel_energies = np.maximum(mel_energies, epsilon())
        mel_energies = np.log(mel_energies)

        feature = np.matmul(self.dct_matrix, mel_energies)

        if self.opts.cepstral_lifter != 0.0:
            feature = feature * self.lifter_coeffs

        if self.opts.use_energy:
            if self.opts.energy_floor > 0.0 and signal_log_energy < self.log_energy_floor:
                signal_log_energy = self.log_energy_floor
            feature[0] = signal_log_energy

        return feature

    def get_mel_banks(self) -> MelBanks:
        if self.mel_banks is None:
            self.mel_banks = MelBanks(self.opts.mel_opts, self.opts.frame_opts)
        return self.mel_banks

    def need_raw_log_energy(self):
        return self.opts.use_energy and self.opts.raw_energy

    def dim(self):
        return self.opts.num_ceps

    def get_frame_extraction_options(self):
        return self.opts.get_frame_options()
