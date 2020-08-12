"""

"""
import os

import numpy as np
import soundfile as sf

from matrix.matrix_io import read_float_mat_vec


class SimpleTextHolder(object):
    def __init__(self):
        pass

    def __call__(self, item: str):
        return item.strip()


class WavHolder(object):
    def __init__(self):
        pass

    def __call__(self, item: str):
        """
        :param item: wav path
        :return:
        """
        if not os.path.isfile(item):
            raise ValueError(f"File {item} do not exist")
        return sf.read(str)


class MatrixHolder(object):
    def __init__(self):
        self.file_handle_dict = dict()

    def _open(self, obj, addr):
        if obj not in self.file_handle_dict:
            self.file_handle_dict[obj] = open(obj, "rb")
        ark_handler = self.file_handle_dict[obj]
        ark_handler.seek(addr)
        return ark_handler

    def __call__(self, script) -> np.ndarray:
        value = script.split(":")
        if len(value) == 1:
            raise ValueError(f"Unsupported scripts address format {script}")
        path, offset = ":".join(value[0:-1]), int(value[-1])
        fd = self._open(path, offset)
        return read_float_mat_vec(fd, direct_access=True)

    def __del__(self):
        for name in self.file_handle_dict:
            self.file_handle_dict[name].close()
