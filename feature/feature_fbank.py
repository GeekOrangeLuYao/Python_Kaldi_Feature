"""
    Extractor FBank Feature

"""

import numpy as np

from feature.feature_window import FrameExtractionOptions
from feature.mel_computations import MelBanksOptions, MelBanks
from base.math_util import epsilon, numeric_limits_float_min


class FbankOptions(object):

    def __init__(self,
                 frame_opts: FrameExtractionOptions,
                 mel_opts: MelBanksOptions,
                 use_energy=False,
                 energy_floor=0.0,
                 raw_energy=True,
                 use_log_fbank=True,
                 use_power=True):
        self.mel_opts = mel_opts
        self.frame_opts = frame_opts

        self.use_energy = use_energy
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.use_log_fbank = use_log_fbank
        self.use_power = use_power

    def get_frame_options(self):
        return self.frame_opts


class FbankComputer(object):

    def __init__(self,
                 opts: FbankOptions):
        self.opts = opts
        self.log_energy_floor = np.log(opts.energy_floor)
        self.mel_bank = None
        self.get_mel_banks()

    def dim(self):
        return self.opts.mel_opts.num_bins + (1 if self.opts.use_energy else 0)

    def get_frame_extraction_options(self):
        return self.opts.frame_opts

    def need_raw_log_energy(self):
        return self.opts.use_energy and self.opts.raw_energy

    def compute(self,
                signal_log_energy,
                signal_frame: np.ndarray) -> np.ndarray:
        assert signal_frame.shape[0] == self.opts.frame_opts.get_padded_window_size()

        mel_banks = self.get_mel_banks()

        if self.opts.use_energy and not self.opts.raw_energy:
            signal_log_energy = np.log(max(np.dot(signal_frame, signal_frame), numeric_limits_float_min()))

        signal_frame = np.fft.fft(signal_frame)
        signal_frame = np.abs(signal_frame) ** 2

        signal_frame_dim = signal_frame.shape[0]
        power_spectrum = signal_frame[:signal_frame_dim // 2 + 1]

        if not self.opts.use_power:
            power_spectrum = np.power(power_spectrum, 0.5)

        # mel_offset = 1 if self.opts.use_energy else 0
        mel_energies = mel_banks.compute(power_spectrum)
        if self.opts.use_log_fbank:
            mel_energies = np.max(mel_energies, epsilon())
            mel_energies = np.log(mel_energies)

        if self.opts.use_energy:
            if self.opts.energy_floor > 0.0 and signal_log_energy < self.log_energy_floor:
                signal_log_energy = self.log_energy_floor
            feature = np.concatenate([np.array(signal_log_energy), mel_energies])
        else:
            feature = mel_energies

        return feature

    def get_mel_banks(self):
        if self.mel_bank is None:
            self.mel_bank = MelBanks(self.opts.mel_opts, self.opts.frame_opts)
        return self.mel_bank
