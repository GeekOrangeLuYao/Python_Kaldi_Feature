"""
    In this project, we just want to use this to read the following:
        * feats.scp -> feature/feature_reader.py
        * wav.scp
    and write the following:
        * feats.ark -> feature/feature_writer.py
        * feats.scp -> feature/feature_writer.py
    Actually, the file should read the files in Kaldi like: xvectors.scp, vad.scp and so on.

    The module do not support the pipe data as we do not use the subprocess lib in python
    We will support the subprocess after finishing the feature-extractor which is more important
    # TODO: Add pipe_or_file data

"""
from typing import Union, Dict, Callable
import os

from util.holder import WavHolder, SimpleTextHolder

Holder = Union[WavHolder, SimpleTextHolder]

__all__ = [
    "SequentialTableReader",  # "SequentialTableScriptReader", "SequentialTableArchiveReader"
    "SequentialTableWriter",  # ""
]


class SequentialTableReader(object):
    def __init__(self,
                 read_specifier,
                 holder: Callable,
                 scp_processor):
        assert read_specifier != ""
        if not os.path.isfile(read_specifier):
            raise RuntimeError(f"Error constructing TableReader: read_specifier is {read_specifier}")
        self.read_specifier = read_specifier
        self.holder = holder
        self.scp_dict = scp_processor(self.read_specifier)
        self.index_keys = list()

    def _load(self, index):
        return self.holder(index)

    def __len__(self):
        return len(self.scp_dict)

    def __iter__(self):
        for key, value in self.scp_dict.items():
            yield key, self._load(value)

    def __contains__(self, item):
        return item in self.scp_dict

    def __getitem__(self, index):
        if len(self.index_keys) != len(self.scp_dict):
            self.index_keys = list(self.scp_dict.keys())

        if type(index) not in [int, str]:
            raise IndexError(f"Unsupported index type: {type(index)}")
        elif type(index) == int:
            if 0 <= index < len(self.scp_dict):
                index = self.index_keys[index]
            else:
                raise KeyError(f"Integer index out of range, {index} vs {len(self.scp_dict)}")
        elif type(index) == str:
            if index not in self.index_keys:
                raise KeyError(f"Missing key {index}")
        return self._load(index)


# class SequentialTableArchiveReader(SequentialTableReader):
#     """
#         ArchiveReader Only for .ark:offset format
#     """
#
#     def __init__(self, rspecifier, holder: Holder):
#         super(SequentialTableArchiveReader, self).__init__(rspecifier, holder)
#
#
# class SequentialTableScriptReader(SequentialTableReader):
#     """
#         ScriptReader for other situations
#     """
#
#     def __init__(self, rspecifier, holder: Holder):
#         super(SequentialTableScriptReader, self).__init__(rspecifier, holder)


class SequentialTableWriter(object):
    def __init__(self, write_specifier, holder: Callable):
        assert write_specifier != ""
        if not os.path.isfile(write_specifier):
            raise RuntimeError(f"Error constructing TableReader: write_specifier is {write_specifier}")
        self.write_specifier = write_specifier
