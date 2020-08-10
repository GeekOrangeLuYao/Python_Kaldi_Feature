"""
Read wav by python, this part will be realized but not used. Abandon it

author: GeekOrangeLuyao
"""

import io
import numpy as np

from base.log_util import throw_boolean

KWaveSampleMax = 32768.0


class WaveHeaderReadGofer(object):
    def __init__(self,
                 istream,
                 swap: bool = False,
                 tag: str = "") -> None:
        self.istream = istream
        self.swap = swap
        self.tag = tag

    def Expect4ByteTag(self, expected: str) -> None:
        actual = bytes.decode(self.istream.read(4))
        throw_boolean(actual == expected, f"WaveData: expected {expected} , got {actual}")
        self.tag = actual

    def Read4ByteTag(self):
        self.tag = bytes.decode(self.istream.read(4))

    def ReadUint32(self):
        result = self.istream.read(4)
        if self.swap:
            # TODO: check the function KALDI_SWAP4
            pass
        # TODO: let the result to a int
        return result

    def ReadUint16(self):
        result = self.istream.read(2)
        if self.swap:
            pass
        return result


def WriteUint32(ostream, i):
    ostream.write(i)


def WriteUint16(ostream, i):
    ostream.write(i)


class WaveInfo(object):

    def __init__(self,
                 samp_freq=0,
                 samp_count=0,
                 num_channels=0,
                 reverse_bytes=0) -> None:
        self.samp_freq = samp_freq
        self.samp_count = samp_count
        self.num_channels = num_channels
        self.reverse_bytes = reverse_bytes

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

    def read(self, istream):
        reader = WaveHeaderReadGofer(istream)
        reader.Read4ByteTag()

        if reader.tag == "RIFF":
            self.reverse_bytes = False
        elif reader.tag == "RIFX":
            self.reverse_bytes = True
        else:
            raise RuntimeError(f"WaveData: expected RIFF or RIFX, got {reader.tag}")

        reader.swap = self.reverse_bytes

        riff_chunk_size = reader.ReadUint32()
        reader.Expect4ByteTag("WAVE")
        riff_chunk_read = 4

        reader.Read4ByteTag()
        riff_chunk_read += 4

        while reader.tag == "fmt ":
            filler_size = reader.ReadUint32()
            riff_chunk_read += 4
            for i in range(filler_size):
                istream.get()

            riff_chunk_read += filler_size
            reader.Read4ByteTag()
            riff_chunk_read += 4

        assert reader.tag == "fmt "
        subchunk1_size = reader.ReadUint32()
        audio_format = reader.ReadUint16()
        self.num_channels = reader.ReadUint16()

        sample_rate = reader.ReadUint32()
        byte_rate = reader.ReadUint32()
        block_align = reader.ReadUint16()
        bits_per_sample = reader.ReadUint16()

        self.samp_freq = float(sample_rate)
        fmt_chunk_read = 16

        # TODO
        raise NotImplementedError


class WaveData(object):
    def __init__(self,
                 data: np.ndarray,
                 samp_freq=0.0) -> None:
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
        raise NotImplementedError

    def clear(self):
        # Kaldi: WaveData::Clear
        # TODO
        raise NotImplementedError


class WaveHolder(object):
    # We won't use Holder like the Kaldi do!

    def __init__(self, source):
        self.source = source

    def value(self):
        raise NotImplementedError
