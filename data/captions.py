import io
import os
import json
import base64
import concurrent
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from typing import Optional

import re
from geopy.geocoders import Nominatim


def parse_coordinates(filename):
    match = re.match(r"NASADEM_HGT_([ns])(\d+)([we])(\d+)\.tif", filename)
    if match:
        lat_sign, lat, lon_sign, lon = match.groups()
        latitude = -int(lat) if lat_sign == 's' else int(lat)
        longitude = -int(lon) if lon_sign == 'w' else int(lon)
        return latitude, longitude
    else:
        raise ValueError("Filename format not recognized")


def get_region_info(latitude, longitude):
    geolocator = Nominatim(user_agent="mincy")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    return location.raw if location else None


def save_region_info(file_list, meta_dir):
    for filename in tqdm(file_list):
        latitude, longitude = parse_coordinates(filename)
        region_info = get_region_info(latitude, longitude)
        data = {}

        if region_info:
            if region_info["display_name"]:
                data["display_name"] = region_info["display_name"]
            else:
                print("No display name information found")

            if region_info["address"] and region_info["address"]["country"]:
                data["country"] = region_info["address"]["country"]
            else:
                print("No country information found")
        else:
            print("No regional information found")

        with open(f"{meta_dir}/{filename.replace('tif', 'json')}", "w+",  encoding='utf8') as file:
            json.dump(data, file, ensure_ascii=False)


def open_meta(filename):
    with open(f"./terrain_metas/{filename.replace('tif', 'json')}", "r", encoding='utf8') as file:
        return json.load(file)


client = OpenAI()


def encode_image(image_path, size=(256, 256)):
    with open(image_path, "rb") as image_file:
        with Image.open(image_file) as img:
            img = img.resize(size)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')


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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    caption = response.choices[0].message.content
    output_tokens = response.usage.completion_tokens
    input_tokens = response.usage.prompt_tokens
    cost = 0.15 * input_tokens / 1e6 + 0.6 * output_tokens / 1e6
    return {"caption": caption, "cost": cost}


def open_meta(filename, meta_dir):
    with open(f"{meta_dir}/{filename.replace('tif', 'json')}", "r", encoding='utf8') as file:
        return json.load(file)


def get_prompt(filename: str, meta_dir):
    meta = open_meta(filename.split("_slice")[0] + ".tif", meta_dir)

    if meta.get("display_name"):
        name_str = f"Location name: {meta['display_name']} "
    else:
        name_str = ""

    if meta.get("country"):
        country_str = f"Country: {meta['country']} "
    else:
        country_str = ""

    if name_str + country_str != "":
        info_str = "Some information about the location: " + name_str + country_str
    else:
        info_str = ""

    prompt = f"""You are tasked with captioning geographical maps to distill their scenic properties.
                {info_str}

                Provide a description of the terrain visible on the image with vivid geomorphic detail.
                Avoid specifying colors, rather talk about the geomorphic elements and elevation.
                Remember: the goal is to specify as many artifacts as possible with terminology from geography.
                The geographical names in the captions should be translated to English.
                Do not use more than two sentences."""

    return prompt


def process_image(filename, image_dir, meta_dir, target_dir):
    prompt = get_prompt(filename, meta_dir)
    filepath = f"{image_dir}/{filename}"

    dem_caption_obj = get_dem_caption(filepath, prompt)
    cost = dem_caption_obj["cost"]

    base_filename = filename.replace('.png', '')
    new_filename = f"{base_filename}.json"

    with open(f"{target_dir}/{new_filename}", 'w', encoding="utf8") as f:
        json.dump(dem_caption_obj, f, ensure_ascii=False)

    return cost


def caption_images(image_dir: str, meta_dir: str, target_dir: str, limit: Optional[int] = None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    exclude_images = [caption.replace("json", "png")
                      for caption in os.listdir(target_dir)]
    filenames = [filename for filename in os.listdir(
        image_dir) if filename not in exclude_images]
    if limit is not None:
        filenames = filenames[:limit]

    total_cost = 0

    with tqdm(total=len(filenames), desc=f"Total cost: {total_cost:.2f}") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_filename = {executor.submit(
                process_image, filename, image_dir, meta_dir, target_dir): filename for filename in filenames}

            for future in concurrent.futures.as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    cost = future.result()
                    total_cost += cost
                except Exception as exc:
                    pass
                pbar.set_description(f"Total cost: {total_cost:.2f}")
                pbar.update(1)
