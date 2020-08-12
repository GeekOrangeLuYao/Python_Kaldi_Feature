"""
    May be it can use multi-process to save the archive

"""

import os
import random
import time


class ArchiveProvider(object):

    def __init__(self, ark_path, split_num, feature_name="feature"):
        self.scp_list = list()
        self.split_num = split_num
        if not os.path.exists(ark_path):
            os.makedirs(ark_path)
        self.ark_pool = [os.path.join(ark_path, f"{feature_name}_{i + 1}.ark") for i in range(split_num)]
        random.seed(time.time())

    def record(self, utt_id, utt_addr):
        self.scp_list.append((utt_id, utt_addr))

    def provide(self):
        return self.ark_pool[random.randint(0, self.split_num - 1)]

    def save(self, scp_file):
        with open(scp_file, "w") as fw:
            for utt_id, utt_addr in self.scp_list:
                fw.write(f"{utt_id} {utt_addr}\n")
            fw.flush()
        print(f"Save to file scp_file {scp_file}")