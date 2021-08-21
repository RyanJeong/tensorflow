* Codes:
    * Learning rate를 0.01로 하면 cost가 발산함
    * `1e-5`부터 수렴하기 시작
    * `1e-6`에서 loss가 가장 좋게 나옴
```python
import tensorflow as tf
import numpy as np

# 4 Multi-variable Linear Regression
x_data = [[73.,80.,75.],
          [93.,88.,93.],
          [89.,91.,90.],
          [96.,98.,100.],
          [73.,66.,70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
tf.model = tf.keras.Sequential()

# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1,input_dim=3))

# SGD == Standard Gradient Descendent, lr == learning rate
sgd = tf.keras.optimizers.SGD(learning_rate=1e-6)

# mse == mean squared error, 1/m*sig(y'-y)^2
tf.model.compile(loss='mse',optimizer=sgd)

# prints summary of the model to the terminal
tf.model.summary()

# fit() executes training
tf.model.fit(x_data,y_data,epochs=100)

# predict() returns predicted value
y_predict = tf.model.predict(np.array([[72.,93.,90.]]))
print(y_predict)
```

* Result:
```text
Epoch 1/100
1/1 [==============================] - 0s 154ms/step - loss: 26508.9688
Epoch 2/100
1/1 [==============================] - 0s 1ms/step - loss: 24227.0859

...

Epoch 99/100
1/1 [==============================] - 0s 1ms/step - loss: 7.1513
Epoch 100/100
1/1 [==============================] - 0s 1ms/step - loss: 6.8148
[[150.57655]]
```