import os

from util.holder import MatrixHolder
from util.provider import ArchiveProvider


class FeatureWriter(object):

    def __init__(self,
                 data_path,
                 split_num):
        self.data_path = data_path
        self.holder = MatrixHolder(mode="wb")
        self.writer = ArchiveProvider(data_path, split_num=split_num)

    def write(self, utt_id, utt_matrix):
        record = self.holder.write(self.writer.provide(), utt_id, utt_matrix)
        self.writer.record(utt_id, record)

    def flush(self):
        scp_path = os.path.join(self.data_path, "feats.scp")
        self.writer.save(scp_path)

    def __del__(self):
        self.flush()
        del self.holder