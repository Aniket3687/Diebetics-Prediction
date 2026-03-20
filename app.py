from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("diabetes_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])

    if prediction[0] == 1:
        result = "Diabetic"
    else:
        result = "Not Diabetic"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)