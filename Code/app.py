
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =====================================================
# 1. INITIALIZE FASTAPI
# =====================================================

app = FastAPI(
    title="Airline Passenger Satisfaction Prediction API",
    description="Predict passenger satisfaction using a trained ANN model.",
    version="1.0.0"
)


# =====================================================
# 2. LOAD SAVED MODEL AND PREPROCESSING FILES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

try:
    model = tf.keras.models.load_model(
        BASE_DIR / "ann_model.keras"
    )

    scaler = joblib.load(
        BASE_DIR / "scaler.pkl"
    )

    selected_features = joblib.load(
        BASE_DIR / "selected_features.pkl"
    )

    encoders = joblib.load(
        BASE_DIR / "label_encoders.joblib"
    )

except Exception as error:
    raise RuntimeError(
        f"Could not load saved files: {error}"
    ) from error


# Confirm the saved model expects the selected feature count.
if len(selected_features) != scaler.n_features_in_:
    raise RuntimeError(
        "The selected feature count does not match the scaler."
    )

if model.input_shape[-1] != len(selected_features):
    raise RuntimeError(
        "The selected feature count does not match the ANN input."
    )


# =====================================================
# 3. PASSENGER INPUT SCHEMA
# =====================================================

class PassengerInput(BaseModel):

    gender: str = Field(description="Male or Female")
    customer_type: str = Field(
        description="Loyal Customer or disloyal Customer"
    )
    age: int = Field(ge=0, le=120)
    type_of_travel: str = Field(
        description="Business travel or Personal Travel"
    )
    passenger_class: str = Field(
        description="Business, Economy, or Economy Plus"
    )

    flight_distance: float = Field(ge=0)

    inflight_wifi_service: int = Field(ge=0, le=5)
    departure_arrival_time_convenient: int = Field(ge=0, le=5)
    ease_of_online_booking: int = Field(ge=0, le=5)
    gate_location: int = Field(ge=0, le=5)
    food_and_drink: int = Field(ge=0, le=5)
    online_boarding: int = Field(ge=0, le=5)
    seat_comfort: int = Field(ge=0, le=5)
    inflight_entertainment: int = Field(ge=0, le=5)
    on_board_service: int = Field(ge=0, le=5)
    leg_room_service: int = Field(ge=0, le=5)
    baggage_handling: int = Field(ge=0, le=5)
    checkin_service: int = Field(ge=0, le=5)
    inflight_service: int = Field(ge=0, le=5)
    cleanliness: int = Field(ge=0, le=5)

    departure_delay_in_minutes: float = Field(ge=0)
    arrival_delay_in_minutes: float = Field(ge=0)


# =====================================================
# 4. LABEL ENCODING HELPER
# =====================================================

def find_encoded_value(column, value):

    mapping = encoders.get(column)

    if not isinstance(mapping, dict):
        raise ValueError(
            f"Encoding mapping for '{column}' was not found "
            "or is not a category-to-integer dictionary."
        )

    for label, encoded_value in mapping.items():

        if str(label).strip().lower() == str(value).strip().lower():
            return int(encoded_value)

    valid_values = ", ".join(
        str(label) for label in mapping.keys()
    )

    raise ValueError(
        f"Invalid value '{value}' for {column}. "
        f"Accepted values: {valid_values}"
    )


# =====================================================
# 5. NORMALIZE PASSENGER CLASS
# =====================================================

def normalize_class(value):

    class_mapping = {
        "business": "Business",
        "economy": "Economy",
        "eco": "Economy",
        "economy plus": "Economy Plus",
        "eco plus": "Economy Plus"
    }

    key = str(value).strip().lower()

    if key not in class_mapping:
        raise ValueError(
            "Class must be Business, Economy, or Economy Plus."
        )

    return class_mapping[key]


# =====================================================
# 6. HOME ENDPOINT
# =====================================================

@app.get("/")
def home():

    return {
        "message": "Airline Passenger Satisfaction Prediction API",
        "documentation": "/docs",
        "prediction_endpoint": "/predict"
    }


# =====================================================
# 7. PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
def predict_satisfaction(passenger: PassengerInput):

    try:

        data = passenger.model_dump()

        # ---------------------------------------------
        # Encode categorical columns
        # ---------------------------------------------

        gender_encoded = find_encoded_value(
            "Gender", data["gender"]
        )

        customer_type_encoded = find_encoded_value(
            "Customer Type", data["customer_type"]
        )

        travel_type_encoded = find_encoded_value(
            "Type of Travel", data["type_of_travel"]
        )

        passenger_class = normalize_class(
            data["passenger_class"]
        )

        # ---------------------------------------------
        # Feature engineering
        # Formulas must match your training notebook
        # ---------------------------------------------

        digital_experience_score = np.mean([
            data["ease_of_online_booking"],
            data["online_boarding"],
            data["inflight_wifi_service"]
        ])

        comfort_score = np.mean([
            data["seat_comfort"],
            data["leg_room_service"],
            data["on_board_service"]
        ])

        service_experience_score = np.mean([
            data["departure_arrival_time_convenient"],
            data["checkin_service"],
            data["gate_location"],
            data["inflight_service"],
            data["inflight_entertainment"],
            data["baggage_handling"],
            data["cleanliness"],
            data["food_and_drink"]
        ])

        # ---------------------------------------------
        # Create model input features
        # Names match selected_features.pkl
        # ---------------------------------------------

        feature_row = {

            "Class_Business": int(
                passenger_class == "Business"
            ),

            "Online Boarding":
                data["online_boarding"],

            "Comfort Score":
                comfort_score,

            "Type of Travel":
                travel_type_encoded,

            "Class_Economy": int(
                passenger_class == "Economy"
            ),

            "In-flight Entertainment":
                data["inflight_entertainment"],

            "Digital Experience Score":
                digital_experience_score,

            "Service Experience Score":
                service_experience_score,

            "Seat Comfort":
                data["seat_comfort"],

            "On-board Service":
                data["on_board_service"],

            "Leg Room Service":
                data["leg_room_service"],

            "Cleanliness":
                data["cleanliness"],

            "Flight Distance":
                data["flight_distance"],

            "In-flight Wifi Service":
                data["inflight_wifi_service"],

            "Baggage Handling":
                data["baggage_handling"],

            # Additional one-hot column created for consistency.
            # It is not passed to the model because it was not selected.
            "Class_Economy Plus": int(
                passenger_class == "Economy Plus"
            )
        }

        # ---------------------------------------------
        # Align columns with saved feature selection
        # ---------------------------------------------

        missing_features = [
            feature
            for feature in selected_features
            if feature not in feature_row
        ]

        if missing_features:
            raise ValueError(
                "Missing model input features: "
                + ", ".join(missing_features)
            )

        input_df = pd.DataFrame([feature_row])

        # Exact same order used during model training.
        input_df = input_df[selected_features]

        # ---------------------------------------------
        # Scale input
        # ---------------------------------------------

        scaled_input = scaler.transform(input_df)

        # ---------------------------------------------
        # Predict with ANN
        # ---------------------------------------------

        probability = float(
            np.asarray(
                model.predict(scaled_input, verbose=0)
            ).reshape(-1)[0]
        )

        predicted_class = int(probability >= 0.5)

        # ---------------------------------------------
        # Decode predicted Satisfaction label
        # ---------------------------------------------

        target_mapping = encoders.get("Satisfaction")

        if not isinstance(target_mapping, dict):
            raise ValueError(
                "Satisfaction mapping was not found in "
                "label_encoders.joblib."
            )

        label_mapping = {
            int(encoded_value): str(label)
            for label, encoded_value in target_mapping.items()
        }

        predicted_label = label_mapping.get(
            predicted_class, "Unknown"
        )

        return {
            "prediction": predicted_label,
            "predicted_class": predicted_class,
            "satisfaction_probability": round(probability, 4),
            "class_used": passenger_class,
            "features_used": len(selected_features)
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )
