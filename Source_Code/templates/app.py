import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import io

app = Flask(__name__)

# 🔹 Load the TensorFlow Models
MODEL_PATH = r"C:\Users\LENOVO\flask_project\model\alzheimer_cnn_model.h5"
VALIDATION_MODEL_PATH = r"C:\Users\LENOVO\flask_project\model\validation_model_new.h5"
model = tf.keras.models.load_model(MODEL_PATH)
validation_model = tf.keras.models.load_model(VALIDATION_MODEL_PATH)

# 🔹 Define class labels
CLASS_LABELS = ["Non-Demented", "Very-Mild Demented", "Mild Demented", "Moderate Demented"]

# 🔹 Function to preprocess the image
def preprocess_image(image, target_size=(208, 104)):
    image = image.resize(target_size)  # Resize to model input size
    image = np.array(image.convert("L")) / 255.0  # Convert to grayscale & normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# 🔹 Function to preprocess image for validation
def preprocess_validation_image(image, target_size=(128, 128)):
    image = image.resize(target_size)  # Resize to validation model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# 🔹 API route to predict image classification
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]  # Get uploaded image
        image = Image.open(io.BytesIO(file.read()))  # Open image
        processed_image = preprocess_image(image)  # Preprocess the image

        # 🔹 Get model prediction
        output = model.predict(processed_image)
        predicted_class = np.argmax(output)  # Get index of max probability
        confidence = np.max(output)  # Get highest confidence score
        
        return jsonify({
            "prediction": CLASS_LABELS[predicted_class],
            "confidence": float(confidence)  # Convert NumPy float to Python float
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})
    

# 🔹 Function to preprocess image for validation
def preprocess_validation_image(image, target_size=(255, 255)):
    image = image.resize(target_size)  # Resize to validation model input size
    image = image.convert("RGB")  # Ensure 3-channel RGB format
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@app.route("/validate_mri", methods=["POST"])
def validate_mri():
    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_validation_image(image)

        output = validation_model.predict(processed_image)
        is_brain_mri = output[0][0] < 0.5  # Assuming binary classification

        return jsonify({"is_mri": bool(is_brain_mri)})

    except Exception as e:
        return jsonify({"error": str(e)})


# 🔹 Route for the symptom checker form
@app.route("/")
def symptom_form():
    return render_template("form.html")  # Ensure "form.html" is the symptom checker page

# 🔹 Route to handle symptom form submission
@app.route("/submit_form", methods=["POST"])
def submit_form():
    symptoms = request.form.getlist("symptom")  # Get checked symptoms

    if len(symptoms) > 3:
        # More than 3 symptoms → Redirect to validation page
        return redirect(url_for("validate_page"))
    else:
        # 3 or fewer symptoms → Show alert and stay on the form page
        return '''
        <script>
            alert("You're alright! No need to upload an image. Just take care of your health with:\n\n" +
                  "- Staying mentally active\n" +
                  "- Eating a balanced diet\n" +
                  "- Getting regular exercise");
            window.location.href = "/";
        </script>
        '''

# 🔹 Route to display the MRI validation page
@app.route("/validate")
def validate_page():
    return render_template("validate.html")  # Ensure "validate.html" is the validation page

# 🔹 Route to display the MRI image upload page
@app.route("/upload")
def upload_page():
    return render_template("index.html")  # Ensure "index.html" is the upload page

if __name__ == "__main__":
    app.run(debug=True)
