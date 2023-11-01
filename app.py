# Flask Imports
import os
from flask import Flask, request, render_template, jsonify, session
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

import secrets

# ------------------Flask App Configuration------------------
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

load_dotenv(find_dotenv())
CORS(app)
app.config.from_mapping(config)
cache = Cache(app)
talisman = Talisman(
    app,
    content_security_policy=csp,
    content_security_policy_nonce_in=["script-src", "style-src"],
)


# ------------------Model Config and Helper Functions------------------
states = ["MA", "NY"]
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
        try:
            img_src = (
                f"{os.environ.get('IMAGE_SOURCE')}{state}/{app.config['images'][key]}"
            )
        except KeyError:
            img_src = "data:image/jpeg;base64,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"
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
@app.route("/classified-points")
def classified_points():
    query = request.args.get("query")
    thresh = request.args.get("thresh")
    max_points = request.args.get("k")
    state = request.args.get("state")
    prev_query = session.get("prev_query")
    print(f"Previous Query: {prev_query}")
    print(f"Current Query: {query}")
    if prev_query == query:
        thresh = thresh
    elif thresh is None or prev_query != query:
        thresh = get_threshold_from_query(query)
    thresh = float(thresh)
    session["prev_query"] = query

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
