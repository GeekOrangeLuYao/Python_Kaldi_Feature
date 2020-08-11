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
    "SequentialTableReader", "SequentialTableScriptReader", "SequentialTableArchiveReader"
]


def parse_scp(scp_path,
              value_processor=lambda x: x,
              num_tokens=2,
              restrict=True) -> Dict:
    scp_dict = dict()
    line = 0

    with open(scp_path, "r") as f:
        for raw_line in f:
            scp_tokens = raw_line.strip().split()
            line += 1
            if scp_tokens[-1] == "|":
                key, value = scp_tokens[0], " ".join(scp_tokens[1:])
            else:
                token_len = len(scp_tokens)
                if num_tokens >= 2 and token_len != num_tokens or restrict and token_len < 2:
                    raise RuntimeError(f"For {scp_path}, format error in line[{line:d}]: {raw_line}")
                if num_tokens == 2:
                    key, value = scp_tokens
                else:
                    key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError(f"Duplicate key {key} exists in {scp_path}")
            scp_dict[key] = value_processor(value)
    return scp_dict


class SequentialTableReader(object):
    def __init__(self, rspecifier, holder: Callable):
        assert rspecifier != ""
        if not os.path.isfile(rspecifier):
            raise RuntimeError(f"Error constructing TableReader: rspecifier is {rspecifier}")
        self.rspecifier = rspecifier
        self.holder = holder
        self.scp_dict = dict()
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


class SequentialTableArchiveReader(SequentialTableReader):
    """
        ArchiveReader Only for .ark:offset format
    """

    def __init__(self, rspecifier, holder: Holder):
        super(SequentialTableArchiveReader, self).__init__(rspecifier, holder)


class SequentialTableScriptReader(SequentialTableReader):
    """
        ScriptReader for other situations
    """

    def __init__(self, rspecifier, holder: Holder):
        super(SequentialTableScriptReader, self).__init__(rspecifier, holder)

        self.scp_dict = parse_scp(rspecifier)


class SequentialTableWriter(object):
    def __init__(self, wspecifier, holder: Callable):
        assert wspecifier != ""
        if not os.path.isfile(wspecifier):
            raise RuntimeError(f"Error constructing TableReader: wspecifier is {wspecifier}")
        self.wspecifier = wspecifier
