import tensorflow as tf
import numpy as np

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]
tf.model = tf.keras.Sequential()

# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1,input_dim=1))

# SGD == Standard Gradient Descendent, lr == learning rate
sgd = tf.keras.optimizers.SGD(learning_rate=0.1) 

# mse == mean squared error, 1/m*sig(y'-y)^2
tf.model.compile(loss='mse',optimizer=sgd)

# prints summary of the model to the terminal
tf.model.summary()

# fit() executes training
tf.model.fit(x_train,y_train,epochs=200)

# predict() returns predicted value
y_predict = tf.model.predict(np.array([5,4]))
print(y_predict)