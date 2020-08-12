import os

from util.table import SequentialTableReader
from util.holder import WavHolder
from util.processor import ScriptProcessor


class WavReader(object):
    def __init__(self, data_path):
        wav_scp_file = os.path.join(data_path, "wav.scp")
        if not os.path.isfile(wav_scp_file):
            raise RuntimeError(f"wav.scp: {wav_scp_file} do not exist")
        self.wav_scp_file = wav_scp_file
        self.holder = WavHolder()
        self.script_processor = ScriptProcessor(scp_path=self.wav_scp_file)
        self.reader = SequentialTableReader(read_specifier=self.wav_scp_file,
                                            holder=self.holder,
                                            scp_processor=self.script_processor)

    def __len__(self):
        return len(self.reader)

    def __iter__(self):
        for key, value in self.reader:
            yield key, value
