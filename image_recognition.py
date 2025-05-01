import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Path to the dataset (adjust if needed)
data_dir = "garbage-classification/Garbage-classification/Garbage-classification"

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Get number of classes from dataset
num_classes = 6

# Load pre-trained MobileNetV2 without the top classifier layers
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base for feature extraction

# Build model using Functional API
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

labels = [label for _, label in train_ds]  # Extract labels from training set
labels = np.concatenate([y.numpy() for y in labels])
weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the top layer
model.fit(train_ds, validation_data=val_ds, class_weight=class_weights, epochs=10)
model.save('models/garbage_classification_model.keras')