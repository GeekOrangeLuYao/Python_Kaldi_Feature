"""

"""
from typing import Union, Dict
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

    def __init__(self, rspecifier):
        assert rspecifier != "" and os.path.isfile(
            rspecifier), f"Error constructing TableReader: rspecifier is {rspecifier}"
        self.rspecifier = rspecifier

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class SequentialTableArchiveReader(SequentialTableReader):
    def __init__(self, rspecifier, holder: Holder):
        super(SequentialTableArchiveReader, self).__init__(rspecifier)
        self.holder = holder


class SequentialTableScriptReader(SequentialTableReader):
    def __init__(self, rspecifier, holder: Holder):
        super(SequentialTableScriptReader, self).__init__(rspecifier)
        self.holder = holder

        self.scp_dict = parse_scp(rspecifier, )

    def __len__(self):
        return len(self.scp_dict)

    def __iter__(self):
        for key, value in self.scp_dict.items():
            yield key, self.holder(value)
