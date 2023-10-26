# Flask Imports
import os
from flask import Flask, request, render_template, jsonify
from flask_caching import Cache
from flask_talisman import Talisman
from flask_cors import CORS
from utils import (
    load_images,
    make_response,
    load_model,
    format_loc,
    get_threshold_from_query,
)

# Model Imports
import numpy as np
import torch.nn.functional as F
import torch

# Config Imports
from config import config, csp
from dotenv import load_dotenv, find_dotenv

# ------------------Flask App Configuration------------------
app = Flask(__name__)
load_dotenv(find_dotenv())
CORS(app)
app.config.from_mapping(config)
cache = Cache(app)
# talisman = Talisman(
#     app,
#     content_security_policy=csp,
#     content_security_policy_nonce_in=["script-src", "style-src"],
# )


# ------------------Model Config and Helper Functions------------------
states = ["MA", "NY", "MIN"]
models = {}

for state in states:
    models[state] = load_model(state)

app.config["images"] = load_images()


def classify(query, thresh=0.05, state="MA"):
    print(f"Query: {query}, Threshold: {thresh}, State: {state}")
    feats, locs, device, textmodel, tokenizer = models[state]

    with torch.no_grad():
        textsenc = tokenizer([query], padding=True, return_tensors="pt").to(device)
        class_embeddings = F.normalize(textmodel(**textsenc).text_embeds, dim=-1)

    classprob = feats @ class_embeddings.cpu().numpy().T
    condition = classprob[:, 0] > thresh
    filtered_locs = locs[condition]

    swapped_points = filtered_locs[:, [1, 0]]
    list_of_swapped_points = swapped_points.tolist()
    top_locs = locs[np.argsort(classprob[:, -1])[::-1][:200]]
    tuples_list = []

    for index, loc in enumerate(top_locs):
        key = format_loc(loc)
        img_src = f"{os.environ.get('IMAGE_SOURCE')}{state}/{app.config['images'][key]}"
        tuples_list.append([img_src, loc.tolist(), index])
    return list_of_swapped_points, tuples_list


# ------------------Flask Status Routes------------------
@app.route("/health")
def health():
    return jsonify({"status": "OK"}), 200


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    print(error)
    return f"Internal error: {error}", 500


# ------------------Flask App Routes------------------
@app.route("/")
def index():
    return render_template("index.html")


@cache.cached(timeout=300)
@app.route("/classified-points", methods=["POST"])
def classified_points():
    query = request.args.get("query")
    thresh = request.args.get("thresh")
    max_points = request.args.get("k")
    state = request.args.get("state")
    thresh = get_threshold_from_query(query)

    # Check for Cache Hit
    cache_key = f"{query}_{thresh}_{state}"
    cached_response = cache.get(cache_key)
    if cached_response:
        print("Cache Hit")
        kwargs = {
            "thresh": str(thresh),
            "blue_coords": cached_response[1],
            "top_locs": cached_response[2],
        }
        if max_points is not None:
            kwargs["top_locs"] = kwargs["top_locs"][: int(max_points)]
            return make_response(**kwargs, status_code=200)
        return make_response(**kwargs, status_code=200)

    # Fetch Results
    list_of_blue_points, top_locs = classify(query, thresh, state)

    # Cache Results and Return
    cache.set(cache_key, [thresh, list_of_blue_points, top_locs], timeout=300)
    return make_response(
        thresh=thresh, blue_coords=list_of_blue_points, top_locs=top_locs
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
