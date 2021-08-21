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