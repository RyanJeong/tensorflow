import tensorflow as tf

x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]
tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1,input_dim=2))

# use sigmoid activation for 0~1 problem
tf.model.add(tf.keras.layers.Activation('sigmoid'))

tf.model.compile(loss='mse',optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),metrics=['accuracy'])
tf.model.summary()
history = tf.model.fit(x_data, y_data, epochs=5000)

# Accuracy report
print("Accuracy: ", history.history['accuracy'][-1])