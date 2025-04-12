import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import kagglehub

dataset_url = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 64
img_width = 64

train_ds = tf.keras.utils.image_dataset_from_directory(                             #serve ad esplicitare i dati per l'allenamento e per il testing
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

""" plt.figure(figsize=(10, 10))                                                       #serve a fare il grafico delle prime dieci immagini
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show() """

normalization_layer = tf.keras.layers.Rescaling(1./255)                               #serve a 'normalizzare', e quindi a fare diventare le immagini leggibili
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

AUTOTUNE = tf.data.AUTOTUNE                                      # serve a memorizzare i dati in cache, facendo in modo che il database dopo la prima epoca si salvi
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 6

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),  # meno filtri
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')  # aggiungi softmax finale
])


model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

def representative_dataset_gen():
    for images, _ in train_ds.take(100):  # 100 batch da 32 immagini = fino a 3200 immagini
        for img in images:
            img = tf.image.resize(img, (img_height, img_width))
            img = tf.cast(img, tf.float32) / 255.0  # normalizzazione manuale
            img = tf.expand_dims(img, axis=0)  # aggiungi batch dimensione
            yield [img]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Model size: {len(tflite_model) / 1024:.2f} KB")