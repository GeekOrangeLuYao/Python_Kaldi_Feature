"""

"""


class ScriptProcessor(object):

    def __init__(self, scp_path, **kwargs):
        self.scp_dict = dict()
        self.scp_path = scp_path
        self.parse_scp(**kwargs)

    def __len__(self):
        return len(self.scp_dict)

    def __iter__(self):
        for key, value in self.scp_dict.items():
            yield key, value

    def parse_scp(self,
                  value_processor=lambda x: x,
                  num_tokens=2,
                  restrict=True):
        scp_path = self.scp_path
        line = 0
        with open(scp_path, "r") as f:
            for raw_line in f:
                scp_tokens = raw_line.strip().split()
                line += 1
                if scp_tokens[-1] == "|":
                    key, value = scp_tokens[0], " ".join(scp_tokens[1:])
                else:
                    token_len = len(scp_tokens)
                    if 2 <= num_tokens != token_len or restrict and token_len < 2:
                        raise RuntimeError(f"For {scp_path}, format error in line[{line:d}]: {raw_line}")
                    if num_tokens == 2:
                        key, value = scp_tokens
                    else:
                        key, value = scp_tokens[0], scp_tokens[1:]
                if key in self.scp_dict:
                    raise ValueError(f"Duplicate key \'{key}\' exists in {scp_path}")
                self.scp_dict[key] = value_processor(value)
        return self.scp_dict
