from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route('/')
@app.route('/classified-points', methods=['POST'])
def search():
    # Get argument: search_query
    # Get body: latitudes and longitudes of rectangular bounding box

    search_query = request.args.get('search_term')
    # body = request.get_json()
    # top_left_lat = body['top_left_lat']
    # top_left_long = body['top_left_long']
    # top_right_lat = body['top_right_lat']
    # top_right_long = body['top_right_long']
    # bottom_left_lat = body['bottom_left_lat']
    # bottom_left_long = body['bottom_left_long']
    # bottom_right_lat = body['bottom_right_lat']
    # bottom_right_long = body['bottom_right_long']
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
