import io
import os
import requests
from flask import jsonify
import numpy as np
from transformers import AutoTokenizer, CLIPTextModelWithProjection


def make_response(status_code=200, **kwargs):
    if not kwargs:
        return jsonify(), status_code

    if "error" in kwargs:
        return jsonify(error=kwargs["error"]), status_code

    return jsonify(**kwargs), status_code


def load_model(file_path="model/MA_2020.npz"):
    if os.environ.get("FLASK_ENV") == "development":
        data = np.load(file_path)
    else:
        response = requests.get(os.environ.get("DATA_SOURCE"))
        data = np.load(io.BytesIO(response.content))
    feats = data["feats"]
    locs = data["locs"]
    device = "cpu"
    textmodel = (
        CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    return feats, locs, device, textmodel, tokenizer


def load_images(file_path="model/data.txt"):
    image_dict = dict()
    if os.environ.get("FLASK_ENV") == "development":
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                key = line.split("/")[-1]
                image_dict[key] = line
    else:
        response = requests.get(os.environ.get("IMAGE_SOURCE") + "filelist_MA_2020.txt")
        file_content = io.StringIO(response.text)  # Convert bytes response to string
        for line in file_content:
            line = line.strip()
            key = line.split("/")[-1]
            image_dict[key] = line
    return image_dict


def format_loc(loc):
    point_lon = str(int(np.round(loc[0] * 100000)))
    point_lat = str(int(np.round(loc[1] * 100000)))
    return point_lon + "_" + point_lat + ".jpg"
