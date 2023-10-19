from flask import jsonify
import numpy as np
from transformers import AutoTokenizer, CLIPTextModelWithProjection


def make_response(blue_coords, top_locs, status_code=200, error=None):
    if error is not None:
        return jsonify(error=error), status_code
    return (
        jsonify(blue_coords=blue_coords, top_locs=top_locs),
        status_code,
    )


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


def format_loc(loc):
    point_lon = str(int(np.round(loc[0] * 100000)))
    point_lat = str(int(np.round(loc[1] * 100000)))
    return point_lon + "_" + point_lat + ".jpg"
