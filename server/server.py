from flask import Flask, request, jsonify
import numpy
import util

app = Flask(__name__)


@app.route("/get_location_names", methods=["GET"])
def get_location_names():
    response = jsonify({"locations": util.get_location_names()})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


@app.route("/predict_home_price", methods=["GET", "POST"])
def predict_home_price():
    house_size = float(request.form["house_size"])
    location = request.form["location"]
    beds = int(request.form["beds"])
    baths = int(request.form["baths"])

    response = jsonify(
        {
            "estimated_price": numpy.float64(
                util.get_estimated_price(
                    baths=baths, beds=beds, location=location, house_size=house_size
                )
            )
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()
