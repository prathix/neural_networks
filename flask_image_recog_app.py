from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('models/garbage_classification_model.keras')

# Class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Home route that provides a simple form for testing
@app.route('/')
def home():
    return '''
        <h1>Upload an image for garbage classification</h1>
        <form action="/predict" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Submit</button>
        </form>
    '''

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Open the image file
        img = Image.open(file.stream)
        img = img.resize((224, 224))  # Resize to model input size
        img = img.convert('RGB')
        
        # Make prediction
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        predictions = model.predict(img_array)
        
        # Get the predicted class index
        predicted_class_idx = np.argmax(predictions, axis=-1)

        # Return the prediction result
        return {'prediction': class_names[predicted_class_idx[0]]}

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
