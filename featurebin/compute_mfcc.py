import argparse

parser = argparse.ArgumentParser(description="The tool to compute the mfcc")
parser.add_argument("--data_path", required=True, help="The data_path to extract mfcc, wav.scp should be included")
parser.add_argument("--save_path", required=True,
                    help="The save_path to save the mfcc features, the building details see the comments please")


def main():
    # TODO: 
    # read in options

    # reader wave and call feature function

    # save features
    return


if __name__ == "__main__":
    main()
