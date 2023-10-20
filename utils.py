from flask import jsonify
import numpy as np
from transformers import AutoTokenizer, CLIPTextModelWithProjection


def make_response(status_code=200, **kwargs):
    if not kwargs:
        return jsonify(), status_code

    if "error" in kwargs:
        return jsonify(error=kwargs["error"]), status_code

    return jsonify(**kwargs), status_code


def load_model():
    data = np.load("model/MA_2020.npz")
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


def load_images():
    image_dict = dict()
    with open("model/data.txt", "r") as f:
        for line in f:
            line = line.strip()
            key = line.split("/")[-1]
            image_dict[key] = line
    return image_dict


def format_loc(loc):
    point_lon = str(int(np.round(loc[0] * 100000)))
    point_lat = str(int(np.round(loc[1] * 100000)))
    return point_lon + "_" + point_lat + ".jpg"
