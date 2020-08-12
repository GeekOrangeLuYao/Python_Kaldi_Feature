"""

Warning:
    The `FeatureReader` is not thread-safety
    because we will use a dict to make the number of file-handler limited
    If you use the same `FeatureReader` instance in different process or thread,
    it could cause mistakes!
    (For example, if you use FeatureReader in torch.data.util.Dataset and set the Dataloader.num_workers > 1,
    it will cause runtime error!)
"""

from util.table import SequentialTableReader
from util.holder import MatrixHolder
from util.processor import ScriptProcessor


class FeatureReader(object):

    def __init__(self, feats_scp):
        self.feats_scp = feats_scp
        # define script_processor function
        self.script_processor = ScriptProcessor(scp_path=feats_scp)
        # define holder function
        self.holder = MatrixHolder(mode="rb")
        self.reader = SequentialTableReader(read_specifier=feats_scp,
                                            holder=self.holder,
                                            scp_processor=self.script_processor)

    def __len__(self):
        return len(self.reader)

    def __iter__(self):
        for key, value in self.reader:
            yield key, value

    def __del__(self):
        del self.holder