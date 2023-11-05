from flask import jsonify
import numpy as np
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from sentence_transformers import SentenceTransformer, util
import torch
from config import GCS

if GCS:
    import os
    from google.cloud import storage

    key = "creds.json"
    storage_client = storage.Client.from_service_account_json(key)
    bucket = storage_client.bucket("graft-models")

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

concepts = [
    "plant",
    "water",
    "concept3",
    "concept4",
    "concept5",
]
concept_embeddings = model.encode(concepts, convert_to_tensor=True)
concept_thresholds = {"plant": 0.16, "water": 0.05}


def get_most_similar_concept(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, concept_embeddings)[0]
    most_similar_concept = concepts[torch.argmax(cosine_scores).item()]
    return most_similar_concept


def make_response(status_code=200, **kwargs):
    if not kwargs:
        return jsonify(), status_code

    if "error" in kwargs:
        return jsonify(error=kwargs["error"]), status_code

    return jsonify(**kwargs), status_code


def load_model(state="MA", device="cpu"):
    if os.environ.get("FLASK_ENV") == "production" and GCS:
        print("Downloading model from GCS")
        print(os.environ.get("FLASK_ENV"))
        blob = bucket.blob(f"{state}_2020.npz")
        blob.download_to_filename(f"model/{state}_2020.npz")

    file_path = f"model/{state}_2020.npz"
    data = np.load(file_path)
    feats = np.array(data["feats"])
    locs = np.array(data["locs"])
    textmodel = (
        CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    return feats, locs, device, textmodel, tokenizer


def initialize_models(states):
    models = dict()
    for state in states:
        models[state] = load_model(state)
    return models


def load_images(files=["model/MA_2020.txt"]):
    image_dict = dict()
    for file_path in files:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                key = line.split("/")[-1]
                image_dict[key] = line
    return image_dict


def format_loc(loc):
    point_lon = str(int(np.round(loc[0] * 100000)))
    point_lat = str(int(np.round(loc[1] * 100000)))
    return point_lon + "_" + point_lat + ".jpg"


def get_threshold_from_query(query):
    matched_concept = get_most_similar_concept(query)
    print(f"Matched Concept: {matched_concept}")
    return concept_thresholds.get(matched_concept, 0.05)
