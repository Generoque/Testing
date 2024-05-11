import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.utils import to_categorical

app = Flask(__name__)

app.config['SCAN_FOLDER'] = r'D:\\testing\\Upload_folder'  # Add the folder to scan

model = load_model("D:\\testing\\models\\RAM_98%.h5")  # Load the Keras model
sequence_length = 90

def preprocess_data(csv_file, num_days):
    df = pd.read_csv(csv_file, usecols=[0, 1, 2, 3, 4])
    if 'Event_encoded' not in df.columns:
        df['Event_encoded'] = LabelEncoder().fit_transform(df['Event ID'])

    new_sequence = df['Event_encoded'].values[-sequence_length:]

    for i in range(num_days):
        new_sequence = np.append(new_sequence, 0)

    new_sequence = new_sequence[-sequence_length:]
    new_sequence = new_sequence.reshape(1, sequence_length, 1)

    predicted_probs = model.predict(new_sequence)
    predicted_label = np.argmax(predicted_probs)
    predicted_pid = df['Event ID'].unique()[predicted_label]

    return predicted_pid

def predict_folder(folder_path, num_days):
    predictions = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            predicted_pid = preprocess_data(file_path, num_days)
            predictions.append((filename, predicted_pid))
    return predictions

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/scan", methods=["POST"])
def scan_folder():
    try:
        folder_path = app.config['SCAN_FOLDER']
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        return render_template("index.html", files=csv_files)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: {}".format(e))

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            num_days = int(request.form['days'])

            # Check if the form contains 'days' parameter, indicating the Predict button was clicked
            if 'days' in request.form:
                files = os.listdir(app.config['SCAN_FOLDER'])
                predictions = predict_folder(app.config['SCAN_FOLDER'], num_days)
                return render_template("index.html", files=files, days=num_days, predictions=predictions)

        except Exception as e:
            return render_template("index.html", prediction_text="Error: {}".format(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
