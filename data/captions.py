import os
import json
import base64
from openai import OpenAI
from typing import Optional


client = OpenAI()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_dem_caption(image_path: str, prompt: str):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    print(response.choices[0])


def get_metas(filename: str):
    return ""


def caption_images(image_dir: str, target_dir: str, limit: Optional[int] = None):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if limit is not None:
        filenames = os.listdir(image_dir)[:limit]
    else:
        filenames = os.listdir(image_dir)

    filepaths = [f"{image_dir}{filename}" for filename in filenames]

    for filename, filepath in zip(filenames, filepaths):

        prompt = f"""
            You are tasked with captioning Digital Elevation Maps for training a terrain generation model.
            Provide an accurate description of the different parts of the captured terrain visible on the
            image with vivid geomorphic detail.
            {get_metas(filename)}
        """

        caption = get_dem_caption(filepath, prompt)

        with open(f"{target_dir}/{filename}", 'w') as f:
            json.dump(caption, f)
