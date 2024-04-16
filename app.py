from flask import Flask, render_template, request, flash
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load the model (consider adding try-except for loading errors)
model_path = "model.h5"
loaded_model = tf.keras.models.load_model(model_path)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the uploaded file
            file = request.files['file']

            # Check if a file was uploaded
            if not file:
                flash("No file uploaded!")
                return render_template('upload.html')

            # Read the image
            img = Image.open(file.stream)

            # Convert the image to RGB
            img = img.convert('RGB')

            # Resize the image
            img = img.resize((224, 224))

            # Convert image to array and normalize
            img_array = np.array(img) / 255.0

            # Expand dimensions to match model input shape
            img_input = np.expand_dims(img_array, axis=0)

            # Make prediction
            pred = loaded_model.predict(img_input)

            # Determine the prediction result
            if pred >= 0.5:
                prediction = "Yes"
            else:
                prediction = "No"

            return render_template('result.html', prediction=prediction)
        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            return render_template('upload.html')

if __name__ == '__main__':
    app.secret_key = 'your_secret_key'  # Add a secret key for flash messages (optional)
    app.run(debug=True)
