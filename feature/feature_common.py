from typing import Union

import numpy as np

from feature.feature_fbank import FbankOptions, FbankComputer
from feature.feature_mfcc import MfccOptions, MfccComputer
from feature.feature_window import FeatureWindowFunction

Options = Union[FbankOptions, MfccOptions]



class OfflineFeature(object):

    def __init__(self,
                 feature_computer_type,
                 opts: Options,
                 ) -> None:
        self.feature_computer_type = feature_computer_type
        self.opts = opts
        self.computer = self._build_computer(feature_computer_type)
        self.feature_windows_function = FeatureWindowFunction(self.computer.get_FrameOptions())

    def _build_computer(self, feature_computer_type):
        if feature_computer_type == 'Mfcc':
            return MfccComputer(self.opts)
        elif feature_computer_type == 'Fbank':
            return FbankComputer(self.opts)
        else:
            raise ValueError(f"{feature_computer_type} do not exist")

    def Compute(self,
                wave: np.ndarray,
                vtln_warp) -> np.ndarray:
        raise NotImplementedError

    def ComputeFeatures(self,
                        wave: np.ndarray,
                        sample_freq,
                        vtln_warp) -> np.ndarray:
        raise NotImplementedError

    def get_dim(self):
        return self.computer.get_dim()
