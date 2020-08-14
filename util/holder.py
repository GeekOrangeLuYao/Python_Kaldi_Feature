"""

"""
import os

import numpy as np
import soundfile as sf

from matrix.matrix_io import read_float_mat_vec, write_float_mat_vec, write_token, write_binary_symbol


class SimpleTextHolder(object):
    def __init__(self):
        pass

    def read(self, item: str):
        return item.strip()


class WavHolder(object):
    def __init__(self):
        pass

    def read(self, item):
        if not os.path.isfile(item):
            raise ValueError(f"File {item} do not exist")
        wave, sample_rate = sf.read(item)
        wave *= (1 << 15)
        return wave, sample_rate

    def write(self, item):
        raise NotImplementedError


class MatrixHolder(object):
    def __init__(self, mode):
        self.file_handle_dict = dict()
        assert mode in ["rb", "wb"]
        self.mode = mode

    def _io(self, obj):
        return open(obj, self.mode)

    def _open(self, obj, addr = None):
        if obj not in self.file_handle_dict:
            self.file_handle_dict[obj] = self._io(obj)
        ark_handler = self.file_handle_dict[obj]
        if addr is not None:
            ark_handler.seek(addr)
        # TODO: Maybe not necessary? check it
        # else:
        #     # should to the end of the file if write mode
        #     ark_handler.tell()
        return ark_handler

    def read(self, script) -> np.ndarray:
        value = script.split(":")
        if len(value) == 1:
            raise ValueError(f"Unsupported scripts address format {script}")
        path, offset = ":".join(value[0:-1]), int(value[-1])
        fd = self._open(path, offset)
        return read_float_mat_vec(fd, direct_access=True)

    def write(self, path, utt_id, utt_matrix) -> str:
        fd = self._open(path)
        write_token(fd, utt_id)
        offset = fd.tell()
        write_binary_symbol(fd)
        write_float_mat_vec(fd, utt_matrix)
        return f"{os.path.abspath(path)}:{offset}"

    def __del__(self):
        for name in self.file_handle_dict:
            self.file_handle_dict[name].close()
