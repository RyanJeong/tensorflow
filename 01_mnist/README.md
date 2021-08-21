* Codes:
```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

* Results:
```text
# python3 main.py
Epoch 1/5
1875/1875 [==============================] - 1s 645us/step - loss: 0.3015 - accuracy: 0.9122
Epoch 2/5
1875/1875 [==============================] - 1s 635us/step - loss: 0.1471 - accuracy: 0.9561
Epoch 3/5
1875/1875 [==============================] - 1s 648us/step - loss: 0.1100 - accuracy: 0.9666
Epoch 4/5
1875/1875 [==============================] - 1s 646us/step - loss: 0.0891 - accuracy: 0.9724
Epoch 5/5
1875/1875 [==============================] - 1s 632us/step - loss: 0.0761 - accuracy: 0.9759
313/313 - 0s - loss: 0.0728 - accuracy: 0.9771
```