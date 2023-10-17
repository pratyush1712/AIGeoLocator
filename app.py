from flask import Flask, request, jsonify, render_template
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch

app = Flask(__name__)


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
    texts = [query]
    with torch.no_grad():
        textsenc = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        class_embeddings = F.normalize(textmodel(**textsenc).text_embeds, dim=-1)

    classprob = feats @ class_embeddings.cpu().numpy().T
    condition = classprob[:, 0] > thresh
    filtered_locs = locs[condition]
    class_confidences = classprob[condition, 0]

    swapped_points = filtered_locs[:, [1, 0]]
    list_of_swapped_points = swapped_points.tolist()
    return list_of_swapped_points, class_confidences.tolist()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classified-points", methods=["POST"])
def classified_points():
    query = request.args.get("query")
    thresh = request.args.get("thresh")
    if thresh == None:
        list_of_blue_points, confidences = classify(query)
    else:
        thresh = float(thresh)
        if thresh < -1 or thresh > 1:
            return (
                jsonify(
                    blue_coords=[],
                    confidences=[],
                    error="Threshold must be between -1 and 1",
                ),
                400,
            )
        list_of_blue_points, confidences = classify(query, thresh)

    return (
        jsonify(blue_coords=list_of_blue_points, confidences=confidences),
        200,
    )


if __name__ == "__main__":
    feats, locs, device, textmodel, tokenizer = load_model()
    app.run(host="localhost", port=8080, debug=True)
