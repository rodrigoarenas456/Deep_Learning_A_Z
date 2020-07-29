import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from preprocessing import CNNModel

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                          gpu_options=gpu_options)

if os.path.isfile("CNN.h5"):
    model = tf.keras.models.load_model("CNN.h5")
else:
    CNN = CNNModel()
    model = CNN.model()
    model.fit(x=CNN.train_set, validation_data=CNN.test_set, epochs=25)
    model.save('CNN.h5')
    print(CNN.train_set.class_indices)

# Single image prediction

test_image = image.load_img('./datasets/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
