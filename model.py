import logging
import google.cloud.logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers import setup_logging
cloud_logger = logging.getLogger('cloudLogger')
cloud_logger.setLevel(logging.INFO)
cloud_logger.addHandler(CloudLoggingHandler(cloud_logging.Client()))
cloud_logger.addHandler(logging.StreamHandler())

# Import TensorFlow
import tensorflow as tf
# Import numpy
import numpy as np

#importing data: Y=3X+1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

#designing the model: ML ALGO: neural network : 1LAYER --> AND EACH LAYER HAS ONE NEURON
# The neural network's input is only one value at a time. Hence, the input shape must be [1]
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

#Compile the model:
# specify 2 functions, a loss and an optimizer.
#mean_squared_error for the loss and stochastic gradient descent (sgd) for the optimizer. 
model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())

#train the neural network :
model.fit(xs, ys, epochs=500) #repeat for 500 timmes 

#using the model : Testing:
cloud_logger.info(str(model.predict([10.0])))
