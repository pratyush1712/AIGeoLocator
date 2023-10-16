from flask import Flask, request, jsonify, render_template
from os.path import join, isfile, isdir
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


def load_model(query):
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
    texts = [
       query
    ]
    with torch.no_grad():
        textsenc = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        class_embeddings = F.normalize(textmodel(**textsenc).text_embeds, dim=-1)

    classprob = feats @ class_embeddings.cpu().numpy().T

    thresh = 0.05
    condition = classprob[:, 0] > thresh

    filtered_locs = locs[condition]
    swapped_points = filtered_locs[:, [1, 0]]
    list_of_swapped_points = swapped_points.tolist()
    return list_of_swapped_points


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/classified-points", methods=["POST"])
def classified_points():
    query = request.args.get("query")
    checkbox_checked = request.args.get("limitPointsBool")
    list_of_blue_points = load_model(query)
    if checkbox_checked == "true":
        return jsonify(query=query, blue_coords=random.sample(list_of_blue_points, min(1000, len(list_of_blue_points)))), 200
    else: 
        return jsonify(query=query, blue_coords=list_of_blue_points), 200
    
if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)
