from flask import Flask, request, jsonify, render_template
import random
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classified-points", methods=["POST"])
def classified_points():
    query = request.args.get("query")
    coordinates = request.json

    southWest = coordinates["southWest"]
    northEast = coordinates["northEast"]

    blue_coords = [
        [
            random.uniform(southWest["lat"], northEast["lat"]),
            random.uniform(southWest["lng"], northEast["lng"])
        ] for _ in range(1000)
    ]

    red_coords = [
        [
            random.uniform(southWest["lat"], northEast["lat"]),
            random.uniform(southWest["lng"], northEast["lng"])
        ] for _ in range(10)
    ]

    return jsonify(query=query, blue_coords=blue_coords, red_coords=red_coords), 200


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)
