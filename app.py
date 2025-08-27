from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load trained ML model
model_path = "model.pkl"  # Updated to your renamed model file
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data and convert to float
        Age = float(request.form.get("Age"))
        Sex = float(request.form.get("Sex"))
        ChestPainType = float(request.form.get("ChestPainType"))
        RestingBP = float(request.form.get("RestingBP"))
        Cholesterol = float(request.form.get("Cholesterol"))
        FastingBS = float(request.form.get("FastingBS"))
        RestingECG = float(request.form.get("RestingECG"))
        MaxHR = float(request.form.get("MaxHR"))
        ExerciseAngina = float(request.form.get("ExerciseAngina"))
        Oldpeak = float(request.form.get("Oldpeak"))
        ST_Slope = float(request.form.get("ST_Slope"))

        # Prepare input features
        features = np.array([[
            Age, Sex, ChestPainType, RestingBP, Cholesterol,
            FastingBS, RestingECG, MaxHR, ExerciseAngina,
            Oldpeak, ST_Slope
        ]])

        # Make prediction
        prediction = model.predict(features)

        # Interpret prediction
        result_text = "⚠️ Chance of Heart Disease" if prediction[0] == 1 else "✅ No Heart Disease"

        return render_template("result.html", prediction_text1=result_text)

    except Exception as e:
        return render_template("result.html", prediction_text1=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
