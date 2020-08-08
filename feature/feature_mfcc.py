import numpy as np

import logging

from feature.feature_window import FrameExtractionOptions
from feature.mel_computations import MelBanks, MelBanksOptions
from matrix.matrix_functions import ComputeDctMatrix

class MfccOptions(object):
    def __init__(self,
                 frame_opts: FrameExtractionOptions,
                 mel_opts: MelBanksOptions = MelBanksOptions(num_bins=23)) -> None:
        self.frame_opts = frame_opts
        self.mel_opts = mel_opts

        self.num_ceps = 13
        self.use_energy = True
        self.energy_floor = 0.0
        self.raw_energy = True
        self.cepstral_lifter = 22.0


class MfccComputer(object):

    def __init__(self,
                 opts: MfccOptions) -> None:
        self.opts = opts
        self.mel_energies = np.zeros((opts.mel_opts.num_bins, ))

        num_bins = opts.mel_opts.num_bins
        if opts.num_ceps > num_bins:
            raise ValueError(f"num-ceps {opts.num_ceps} cannot be larger than num-mel-bins {num_bins}.")

