import logging

import numpy as np

from feature.feature_config import OptionsParser


class VadEnergyOptions(object):
    def __init__(self,
                 vad_energy_threshold=5.0,
                 vad_energy_mean_scale=0.5,
                 vad_frames_context=0,
                 vad_proportion_threshold=0.6):
        self.vad_energy_threshold = vad_energy_threshold
        self.vad_energy_mean_scale = vad_energy_mean_scale
        self.vad_frames_context = vad_frames_context
        self.vad_proportion_threshold = vad_proportion_threshold

    def register(self, option_parser: OptionsParser):
        # self.use_energy = option_parser.get("use_energy", False, type_function=np.bool)
        self.vad_energy_threshold = option_parser.get("vad_energy_threshold", 5.0, type_function=np.float)
        self.vad_energy_mean_scale = option_parser.get("vad_energy_mean_scale", 0.5, type_function=np.float)
        self.vad_frames_context = option_parser.get("vad_frames_context", 0, type_function=np.int)
        self.vad_proportion_threshold = option_parser.get("vad_proportion_threshold", 0.6, type_function=np.float)


def compute_vad_energy(opts: VadEnergyOptions, feature: np.ndarray) -> np.ndarray:
    feat_length = feature.shape[0]
    if feat_length == 0:
        logging.warning(f"Empty features")
        return np.array([])
    log_energy = feature[:, 0]
    energy_threshold = opts.vad_energy_threshold

    if opts.vad_energy_mean_scale != 0.0:
        assert opts.vad_energy_mean_scale > 0.0
        energy_threshold += opts.vad_energy_mean_scale * np.sum(log_energy) / feat_length

    assert opts.vad_frames_context >= 0
    assert 0.0 < opts.vad_proportion_threshold < 1.0

    output_voiced = []
    for t in range(feat_length):
        log_energy_data = log_energy[t]
        num_count = 0
        den_count = 0
        context = opts.vad_frames_context
        for t2 in range(t - context, t + context):
            if 0 <= t2 < feat_length:
                den_count += 1
                if log_energy_data[t2] > energy_threshold:
                    num_count += 1

        if num_count >= den_count * opts.vad_proportion_threshold:
            output_voiced.append(1.0)
        else:
            output_voiced.append(0.0)
    return np.array(output_voiced)

