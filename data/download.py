import os
import requests
import zipfile

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def main():
    files_to_download = [
        {'file_id': '1zy3ccXIm_Kbrbwg8V0jS_plWPmkuorTh', 'destination': './data/terrain.zip', 'extract_to': './data/terrain'},
        {'file_id': '1uh6BjRbJy8x2i42ghYtiXVv_bAREwtfH', 'destination': './data/context.zip', 'extract_to': './data/context'},
    ]

    for file_info in files_to_download:
        download_file_from_google_drive(file_info['file_id'], file_info['destination'])
        os.remove(file_info['destination'])

if __name__ == "__main__":
    main()
