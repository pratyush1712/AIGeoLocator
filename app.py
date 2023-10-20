# Flask Imports
from flask import Flask, request, render_template
from flask_caching import Cache
from flask_talisman import Talisman
from flask_cors import CORS
from utils import load_images, make_response, load_model, format_loc

# Model Imports
import numpy as np
import torch.nn.functional as F
import torch

# Config Imports
from config import config, csp

# ------------------Flask App Configuration------------------
app = Flask(__name__)
CORS(app)
app.config.from_mapping(config)
cache = Cache(app)
talisman = Talisman(
    app,
    content_security_policy=csp,
    content_security_policy_nonce_in=["script-src", "style-src"],
)


# ------------------Model Config and Helper Functions------------------
def classify(query, thresh=0.05):
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
        tuples_list.append([app.config["image_dict"][key], loc.tolist(), index])
    return list_of_swapped_points, tuples_list


# ------------------Flask Status Routes------------------
@app.route("/health")
def health():
    return "OK"


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

    # Check for Cache Hit
    cache_key = f"{query}_{thresh}"
    cached_response = cache.get(cache_key)
    if cached_response:
        print("Cache Hit")
        kwargs = {"blue_coords": cached_response[0], "top_locs": cached_response[1]}
        if max_points is not None:
            kwargs["top_locs"] = kwargs["top_locs"][: int(max_points)]
            return make_response(**kwargs, status_code=200)
        return make_response(**kwargs, status_code=200)

    # Fetch Results
    thresh = 0.05 if thresh is None else float(thresh)
    list_of_blue_points, top_locs = classify(query, thresh)

    # Cache Results and Return
    cache.set(cache_key, [list_of_blue_points, top_locs], timeout=300)
    return make_response(blue_coords=list_of_blue_points, top_locs=top_locs)


if __name__ == "__main__":
    feats, locs, device, textmodel, tokenizer = load_model()
    app.config["image_dict"] = load_images()
    app.run(host="0.0.0.0", port=8080, debug=True)
