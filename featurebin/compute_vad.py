import os
import argparse
import logging

import numpy as np

from feature.feature_reader import FeatureReader
from feature.feature_writer import FeatureWriter
from feature.feature_config import OptionsParser
from feature.voice_activity_detection import VadEnergyOptions, compute_vad_energy

parser = argparse.ArgumentParser(description="The tool to compute the vad")
parser.add_argument("--data_path",
                    required=True,
                    help="The data_path to compute vad, feats.scp should be included")
parser.add_argument("--save_path",
                    required=True,
                    help="The save_path to save the vad, the building details see the comments please")
parser.add_argument("--config_file",
                    required=True,
                    help="The config file to extract vad")
parser.add_argument("--config_section",
                    help="The config section, default = [default]")


def main(args):
    data_path = args.data_path
    save_path = args.save_path
    config_file = args.config_file
    config_section = args.config_section

    option_parser = OptionsParser(conf_file=config_file, conf_section=config_section)
    vad_opts = VadEnergyOptions()
    vad_opts.register(option_parser)

    feats_scp = os.path.join(data_path, "feats.scp")
    feature_reader = FeatureReader(feats_scp)
    vad_writer = FeatureWriter(save_path, split_num=1)

    num_done = 0
    num_err = 0
    num_unvoiced = 0
    tot_length = 0.0
    tot_decision = 0.0
    for (utt_id, utt_feat) in feature_reader:
        if utt_feat.shape[0] == 0:
            logging.warning(f"Empty feature matrix for utterance {utt_id}")
            num_err += 1
            continue

        vad_result = compute_vad_energy(vad_opts, utt_feat)
        vad_sum = np.sum(vad_result)
        if vad_sum == 0.0:
            logging.warning(f"No frames were judged voiced for utterance {utt_id}")
            num_unvoiced += 1
        else:
            num_done += 1

        tot_decision += np.sum(vad_result)
        tot_length += vad_result.shape[0]

        if vad_sum != 0.0:
            vad_writer.write(utt_id, vad_result)

    logging.info(f"Applied energy based voice activity detection "
                 f"{num_done} utterances successfully"
                 f"{num_err} had empty features, and "
                 f"{num_unvoiced} were completely unvoiced.")
    logging.info(f"Proportion of voiced frames was"
                 f"{float(tot_decision) / float(tot_length)} over"
                 f"{tot_length} frames")


if __name__ == '__main__':
    args = parser.parse_args()
    logging.info(f"{args}")
    main(args)
