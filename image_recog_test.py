import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/garbage_classification_model.keras')

# Load an image you want to classify (replace with your image path)

img = tf.keras.utils.load_img("test_data/trash_test.jpg", target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

# Make prediction
predictions = model.predict(img_array)

# Get the predicted class index
predicted_class_idx = np.argmax(predictions, axis=-1)

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(predictions)