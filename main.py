import joblib
import pandas as pd
import argparse

# Load model
model = joblib.load("best_model.pkl")

def predict(input_data):
    df = pd.DataFrame([input_data])  # single row
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMW Sales Prediction")
    parser.add_argument("--mileage", type=float, required=True)
    parser.add_argument("--fuel_type", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    
    args = parser.parse_args()
    input_data = {
        "mileage": args.mileage,
        "fuel_type": args.fuel_type,
        "region": args.region
    }
    
    print("Prediction:", predict(input_data))
