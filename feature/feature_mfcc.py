import numpy as np

import logging

from feature.feature_window import FrameExtractionOptions
from feature.mel_computations import MelBanks, MelBanksOptions, ComputeLifterCoeffs
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
        self.srfft = None
        self.mel_energies = opts.mel_opts.num_bins

        num_bins = opts.mel_opts.num_bins
        if opts.num_ceps > num_bins:
            logging.error(
                f"num-ceps cannot be larger than num-mel-bins.  It should be smaller or equal. You provided num-ceps: {opts.num_ceps} and num-mel-bins: {num_bins}")

        dct_matrix = np.zeros((num_bins, num_bins))
        dct_matrix = ComputeDctMatrix(dct_matrix)
        self.dct_matrix = dct_matrix[:opts.num_ceps, :num_bins]
        
        if opts.cepstral_lifter != 0.0:
            self.lifter_coeffs = np.zeros((opts.num_ceps, ))
            self.lifter_coeffs = ComputeLifterCoeffs(opts.cepstral_lifter, self.lifter_coeffs)

        if opts.energy_floor > 0.0:
            # TODO: check Log in Kaldi
            self.log_energy_floor = np.log(opts.energy_floor)
        padded_window_size = opts.frame_opts.get_padded_window_size()

        if ((padded_window_size) & (padded_window_size - 1)) == 0:
            # TODO: SplitRadixRealFft
            self.srfft = np.zeros((padded_window_size, ))

        # The data not used, should be used in the following functions
        self.mel_banks = dict()
        self._get_mel_banks(1.0)

    def get_dim(self) -> int:
        return self.opts.num_ceps

    def need_raw_log_energy(self):
        return (self.opts.use_energy and self.opts.raw_energy)

    def get_FrameOptions(self):
        return self.opts.frame_opts

    def Compute(self,
                signal_log_energy,
                vtln_warp,
                signal_frame: np.ndarray,
                feature: np.ndarray):
        assert signal_frame.shape[0] == self.opts.frame_opts.get_padded_window_size()
        assert feature.shape[0] == self.get_dim()

        mel_banks: MelBanks = self._get_mel_banks(vtln_warp)

        # TODO: finished compute

    def _get_mel_banks(self, vtln_warp):
        res = self.mel_banks[vtln_warp]
        if res is None:
            this_mel_banks = MelBanks(self.opts.mel_opts,
                                      self.opts.frame_opts,
                                      vtln_warp)
            self.mel_banks[vtln_warp] = this_mel_banks
        else:
            this_mel_banks = res[1]

        return this_mel_banks
