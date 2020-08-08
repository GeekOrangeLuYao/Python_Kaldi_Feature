import numpy as np

from feature.feature_window import FrameExtractionOptions
from feature.mel_computations import MelBanksOptions, MelBanks


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
        self.mel_bank = MelBanks(opts.mel_opts, opts.frame_opts)
        # only use the vtln_warp == 1.0, so we will only save one MelBank

    def get_dim(self):
        return self.opts.mel_opts.num_bins + (1 if self.opts.use_energy else 0)

    def get_frame_extraction_options(self):
        return self.opts.frame_opts

    def need_raw_log_energy(self):
        return self.opts.use_energy and self.opts.raw_energy

    def compute(self,
                signal_log_energy,
                signal_frame: np.ndarray) -> np.ndarray:
        assert signal_frame.shape[0] == self.opts.frame_opts.get_padded_window_size()
