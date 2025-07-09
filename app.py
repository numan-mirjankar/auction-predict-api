from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow frontend calls

CATEGORIES = [
    "Electronics", "Antiques", "Art", "Books", "Clothing", "Collectibles",
    "Home & Garden", "Jewelry", "Musical Instruments", "Sports", "Toys",
    "Vehicles", "Other"
]

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Auction Final Bid Prediction API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        category = data.get("category")
        starting_price = float(data.get("starting_price"))
        duration_days = int(data.get("duration_days"))
        bid_count = int(data.get("bid_count"))

        # One-hot encode category
        category_features = [1 if category == cat else 0 for cat in CATEGORIES]
        features = [starting_price, duration_days, bid_count] + category_features

        # Predict
        prediction = model.predict([features])[0]
        return jsonify({"success": True, "predicted_price": round(prediction, 2)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run()
