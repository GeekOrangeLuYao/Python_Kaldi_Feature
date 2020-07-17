import numpy as np

class WaveInfo(object):
    # TODO: use wave or soundfile lib to get this instead of reading bytes
    def __init__(self,
                 samp_freq = 0,
                 samp_count = 0,
                 num_channels = 0,
                 reverse_bytes = 0) -> None:
        self._samp_freq = samp_freq
        self._samp_count = samp_count
        self._num_channels = num_channels
        self._reverse_bytes = reverse_bytes
    
    @property
    def samp_freq(self):
        return self._samp_freq

    @property
    def samp_count(self):
        return self._samp_count
    
    @property
    def num_channels(self):
        return self._num_channels
    
    @property
    def reverse_bytes(self):
        return self._reverse_bytes

    def is_streamed(self) -> bool:
        # Kaldi: WaveInfo::IsStreamed
        return self.samp_count < 0
    
    def get_duration(self):
        # Kaldi: WaveInfo::Duration
        return self.samp_count / self.samp_freq

    def get_block_align(self):
        # Kaldi: WaveInfo::BlockAlign
        return 2 * self.num_channels

    def get_data_bytes(self):
        # Kaldi: WaveInfo::DataBytes
        return self.samp_count * self.get_block_align()

    def read(self):
        raise NotImplementedError

class WaveData(object):
    def __init__(self,
                 data: np.ndarray,
                 samp_freq = 0.0) -> None:
        self._data = data
        self._samp_freq = samp_freq
    
    @property
    def data(self):
        return self._data

    @property
    def samp_freq(self):
        return self._samp_freq

    def read(self):
        raise NotImplementedError

    def write(self):
        raise NotImplementedError

    def get_duration(self):
        # Kaldi: WaveData::Duration
        return self.data.shape[1] / self.samp_freq
    
    def copy_from(self, other):
        # Kaldi: WaveData::CopyFrom
        self.samp_freq = other.samp_freq
        self.data = np.copy(other.data)
    
    def clear(self):
        # Kaldi: WaveData::Clear
        # TODO
        raise NotImplementedError