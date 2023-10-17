from flask import Flask, request, jsonify, render_template
from os.path import join
from os import listdir
from multiprocessing import Pool
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch
import random

app = Flask(__name__)
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

def load_model(query):
    texts = [query]
    with torch.no_grad():
        textsenc = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        class_embeddings = F.normalize(
            textmodel(**textsenc).text_embeds, dim=-1
        )

    classprob = feats @ class_embeddings.cpu().numpy().T

    thresh = 0.05
    condition = classprob[:, 0] > thresh

    filtered_locs = locs[condition]
    class_confidences = classprob[condition, 0]

    swapped_points = filtered_locs[:, [1, 0]]
    list_of_swapped_points = swapped_points.tolist()
    print(class_confidences.tolist()[0:5])
    return list_of_swapped_points, class_confidences.tolist()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classified-points", methods=["POST"])
def classified_points():
    query = request.args.get("query")
    #checkbox_checked = request.args.get("limitPointsBool")
    list_of_blue_points, confidences = load_model(query)

    return jsonify(
            query=query, blue_coords=list_of_blue_points, confidences=confidences
        ), 200

    # if checkbox_checked == "true":
    #     # Limit the number of points
    #     num_points = min(1000, len(list_of_blue_points))
    #     selected_indices = random.sample(range(len(list_of_blue_points)), num_points)
    #     selected_points = [list_of_blue_points[i] for i in selected_indices]
    #     selected_confidences = [confidences[i] for i in selected_indices]

    #     return jsonify(
    #         query=query,
    #         blue_coords=selected_points,
    #         confidences=selected_confidences,
    #     ), 200
    # else:
    #     return jsonify(
    #         query=query, blue_coords=list_of_blue_points, confidences=confidences
    #     ), 200


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)
