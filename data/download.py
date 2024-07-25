import argparse
import os
import shutil
import gdown


def main(unpack=False):
    files_to_download = [
        {'file_id': '1zy3ccXIm_Kbrbwg8V0jS_plWPmkuorTh',
            'destination': './data/elevation.zip'},
        {'file_id': '1uh6BjRbJy8x2i42ghYtiXVv_bAREwtfH',
            'destination': './data/context.zip'},
    ]

    for file_info in files_to_download:
        gdown.download(
            f"https://drive.google.com/uc?id={file_info['file_id']}", file_info["destination"])

        if unpack:
            dest = file_info["destination"].replace(".zip", "")
            shutil.unpack_archive(file_info["destination"], dest)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument('--unpack', action=argparse.BooleanOptionalAction, default=False,
                        help='Unzip downloaded archives')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.unpack)
