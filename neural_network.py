import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/ 255.0, x_test/255.0

Model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)
])

predictions = Model(x_train[:1]).numpy()
predictions

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

Model.compile(optimizer = 'adam', 
              loss = loss_fn,
              metrics = ['accuracy'])

Model.fit(x_train, y_train, epochs = 10)
Model.evaluate(x_train, y_train, verbose = 2)

Model.save('models/handwriting_number_recognition.keras')