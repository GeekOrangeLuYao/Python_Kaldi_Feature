import argparse
import logging

from feature.feature_common import FeatureExtractor
from feature.feature_config import OptionsParser
from feature.feature_writer import FeatureWriter
from audio.wav_reader import WavReader

parser = argparse.ArgumentParser(description="The tool to compute the mfcc")
parser.add_argument("--data_path",
                    required=True,
                    help="The data_path to extract mfcc, wav.scp should be included")
parser.add_argument("--save_path",
                    required=True,
                    help="The save_path to save the mfcc features, the building details see the comments please")
parser.add_argument("--config_file",
                    required=True,
                    help="The config file to extract features")
parser.add_argument("--config_section",
                    help="The config section")


def main(args):
    data_path = args.data_path
    save_path = args.save_path
    config_file = args.config_file
    config_section = args.config_section

    wav_reader = WavReader(data_path)
    option_parser = OptionsParser(conf_file=config_file, conf_section=config_section)
    feature_extractor = FeatureExtractor(feature_type="mfcc", option_parser=option_parser)
    feature_writer = FeatureWriter(save_path, split_num=1)

    for utt_id, (wav, sample_rate) in wav_reader:
        # print(f"waveform:\n{wav}\n")
        result = feature_extractor.compute_features(wav, sample_rate)
        print(f"utt_id = {utt_id}, result.shape = {result.shape}")
        feature_writer.write(utt_id, result)
    feature_writer.flush()


if __name__ == "__main__":
    args = parser.parse_args()
    logging.info(f"{args}")
    main(args)
