import os
import gdown


def main():
    files_to_download = [
        {'file_id': '1zy3ccXIm_Kbrbwg8V0jS_plWPmkuorTh',
            'destination': './data/elevation.zip'},
        {'file_id': '1uh6BjRbJy8x2i42ghYtiXVv_bAREwtfH',
            'destination': './data/context.zip'},
    ]

    for file_info in files_to_download:
        gdown.download(
            f"https://drive.google.com/uc?id={file_info['file_id']}", file_info["destination"])


if __name__ == "__main__":
    main()
