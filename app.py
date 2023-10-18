from flask import Flask, request, jsonify, render_template
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from flask_caching import Cache
import torch

app = Flask(__name__)
cache = Cache(app, config={"CACHE_TYPE": "simple"})


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


def classify(query, thresh=0.05):
    print("function called")
    texts = [query]
    with torch.no_grad():
        textsenc = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        class_embeddings = F.normalize(textmodel(**textsenc).text_embeds, dim=-1)

    classprob = feats @ class_embeddings.cpu().numpy().T
    condition = classprob[:, 0] > thresh
    filtered_locs = locs[condition]

    swapped_points = filtered_locs[:, [1, 0]]
    list_of_swapped_points = swapped_points.tolist()
    top_locs = locs[np.argsort(classprob[:, -1])[::-1][:200]]
    tuples_list = []

    for index, po in enumerate(top_locs):
        key = (
            str(int(np.round(po[0] * 100000)))
            + "_"
            + str(int(np.round((po[1]) * 100000)))
            + ".jpg"
        )
        tuples_list.append([app.config["image_dict"][key], po.tolist(), index])
    return list_of_swapped_points, tuples_list


@app.route("/")
def index():
    return render_template("index.html")


@cache.cached(timeout=300)
@app.route("/classified-points", methods=["POST"])
def classified_points():
    query = request.args.get("query")
    thresh = request.args.get("thresh")
    max_points = request.args.get("k")

    cache_key = f"{query}_{thresh}"
    cached_response = cache.get(cache_key)
    if cached_response:
        print("Cache hit")
        if max_points is not None:
            print("Max points: " + str(max_points))
            return make_response(
                cached_response[0],
                cached_response[1][: int(max_points)],
                status_code=200,
            )
        return make_response(cached_response[0], cached_response[1], status_code=200)
    print("Cache miss")
    if thresh == None:
        list_of_blue_points, top_locs = classify(query)
    else:
        thresh = float(thresh)
        if thresh < -1 or thresh > 1:
            return make_response([], [], 400, "Thresh must be between -1 and 1")
        list_of_blue_points, top_locs = classify(query, thresh)

    cache.set(cache_key, [list_of_blue_points, top_locs], timeout=300)
    return make_response(list_of_blue_points, top_locs, status_code=200)


if __name__ == "__main__":
    feats, locs, device, textmodel, tokenizer = load_model()
    image_dict = {}
    with open("model/data.txt", "r") as f:
        for line in f:
            line = line.strip()
            key = line.split("/")[-1]
            image_dict[key] = line
    app.config["image_dict"] = image_dict

    app.run(host="localhost", port=8080, debug=True)
