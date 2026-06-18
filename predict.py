import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model.keras")

# lấy lại class
class_names = ["apple", "banana", "orange"]  # sửa đúng theo dataset bạn

img = tf.keras.utils.load_img(
    "test.jpg",
    target_size=(100,100)
)

img = tf.keras.utils.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

result = class_names[np.argmax(pred)]

print("Prediction:", result)