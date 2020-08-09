import argparse
import logging

parser = argparse.ArgumentParser(description="The tool to compute the fbank")
parser.add_argument("--data_path",
                    required=True,
                    help="The data_path to extract mfcc, wav.scp should be included")
parser.add_argument("--save_path",
                    required=True,
                    help="The save_path to save the mfcc features, the building details see the comments please")
parser.add_argument("--config_file",
                    help="The config file to extract features")


def main():
    return


if __name__ == '__main__':
    main()
