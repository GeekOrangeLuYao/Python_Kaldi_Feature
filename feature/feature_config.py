"""
    Use .ini file to read config
"""
import os
import configparser as cfp

import numpy as np


class OptionsParser(object):

    def __init__(self,
                 conf_file,
                 conf_section="default"):
        self.config_dict = dict()

        if not os.path.isfile(conf_file):
            raise ValueError(f"Read config file {conf_file} failed!")
        config = cfp.ConfigParser()
        config.read(conf_file)

        assert conf_section in config
        for key in config[conf_section]:
            self.config_dict[key] = config[conf_section][key]
            print(f"key = {key}, value = { config[conf_section][key] }")

    def get(self,
            item,
            default_value,
            type_function):
        if type_function == np.bool:
            return self.config_dict.get(item, default_value) == "True"
        else:
            return type_function(self.config_dict.get(item, default_value))
