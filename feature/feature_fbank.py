

from feature.feature_window import FrameExtractionOptions
from feature.mel_computations import MelBanksOptions

class FbankOptions(object):

    def __init__(self,
                 frame_opts: FrameExtractionOptions,
                 mel_opts: MelBanksOptions = MelBanksOptions(num_bins=23)) -> None:
        self.frame_opts = frame_opts
        self.mel_opts = mel_opts

        self.use_energy = True
        self.energy_floor = 0.0
        self.raw_energy = True
        self.use_log_fbank = True
        self.use_power = True

class FbankComputer(object):
    def __init__(self,
                 opts: FbankOptions) -> None:
        self.opts = opts
        self.srfft = None
