class SequentialTableReader(object):

    def __init__(self,
                 # TODO: support the shell code here
                 scp_file,
                 holder) -> None:
        self.scp_file = scp_file
        self.holder = holder
        self.scp_dict = dict()
        self.index_keys = list(self.scp_dict.keys())

        self._open()

    def _open(self):
        with open(self.scp_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                fb = line.find(" ")  # first blank
                key = line[:fb]
                value = line[fb + 1:]
                if key in self.scp_dict:
                    raise ValueError(f"Duplicate key in your scp_fil {self.scp_file}")
                self.scp_dict[key] = value

    def __len__(self):
        return len(self.scp_dict)

    def __getitem__(self, item):
        if isinstance(item, int):
            assert 0 <= item < len(self), f"Wrong range to index"
            return self.scp_dict[self.index_keys[item]]
        elif isinstance(item, str):
            return self.scp_dict[item]
        else:
            raise ValueError(f"Bad index {item}")

    def __iter__(self):
        for (key, value) in self.scp_dict:
            # TODO: use holder here
            yield key, self.holder(value).value()

    def __repr__(self):
        return f"scp_file = {self.scp_file}, holder = {repr(self.holder)}"
