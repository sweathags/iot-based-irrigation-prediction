# app.py
from flask import Flask, render_template, request, redirect
from joblib import load
import pandas as pd
import firebase_admin
from firebase_admin import credentials,db

# Initialize Firebase (replace with your own credentials)
cred = credentials.Certificate('your-private-key')
app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'your-db-url'
})

app = Flask(__name__, template_folder="templates")

# Load the saved model
loaded_model = load("random_forest_model.joblib")

@app.route('/')
def index():
    # Get data from Firebase using the Firebase Admin SDK
    ref = db.reference('/')
    data = ref.get()

    # Extract specific values (assuming data is a dictionary)
    hum = data.get('Hum')
    temp = data.get('Temp')
    soil = data.get('Soil')

    # Prepare context for the template
    context = {
        'hum': hum,
        'temp': temp,
        'soil': soil
    }

    return render_template('form.html', **context)

@app.route("/", methods=["POST"])
def submit_form():
    prediction=None

    if request.method == "POST":
        # Get input data from the form
        CropType = request.form["CropType"]
        CropDays = request.form["CropDays"]
        SoilMoisture = request.form["SoilMoisture"]
        temperature = request.form["temperature"] 
        Humidity = request.form["Humidity"]

        # Create new_data DataFrame
        new_data = pd.DataFrame({
            'CropType': [CropType],
            'CropDays': [CropDays],
            'SoilMoisture': [SoilMoisture],
            'temperature': [temperature],
            'Humidity': [Humidity]
        })

        # Make predictions
        predictions = loaded_model.predict(new_data)

        # Assign prediction
        prediction = predictions[0]

        # Update pump state in Firebase (store as integer)
        db = firebase_admin.db.reference()
        db.child('pump').set(str(prediction))

    # Render the form template with prediction value embedded
    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
