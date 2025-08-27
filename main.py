import joblib
import pandas as pd
import argparse
import json

# Load model
model = joblib.load("models/best_model.pkl")

# (Optional) Load metadata if you saved it
try:
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
        print(f"Loaded model: {metadata.get('best_model', 'Unknown')}")
except FileNotFoundError:
    print("No metadata file found. Using default model.")

# Example feature names (⚠️ must match your training dataset!)
FEATURES = ["mileage", "fuel_type", "region"]

def predict(input_data: dict):
    """Run prediction on one sample input_data dict"""
    df = pd.DataFrame([input_data])  # create a dataframe with one row
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMW Price / Sales Predictor")
    parser.add_argument("--mileage", type=float, required=True, help="Mileage of the car")
    parser.add_argument("--fuel_type", type=str, required=True, help="Fuel type (e.g. Petrol, Diesel, Hybrid)")
    parser.add_argument("--region", type=str, required=True, help="Region (e.g. Europe, Asia, US)")

    args = parser.parse_args()
    
    # Create input dict
    input_data = {
        "mileage": args.mileage,
        "fuel_type": args.fuel_type,
        "region": args.region
    }

    result = predict(input_data)
    print("Prediction:", result)
