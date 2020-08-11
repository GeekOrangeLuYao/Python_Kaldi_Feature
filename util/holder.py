"""

"""
import os
import soundfile as sf


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
