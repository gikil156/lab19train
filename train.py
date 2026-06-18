import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. LOAD DATASET
# =========================

IMG_SIZE = (100, 100)
BATCH_SIZE = 32

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = dataset.class_names
print("Classes:", class_names)

# =========================
# 2. CHIA TRAIN / TEST
# =========================

train_size = int(0.8 * len(dataset))

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

# tối ưu tốc độ
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# =========================
# 3. CNN MODEL
# =========================

model = tf.keras.Sequential([

    # chuẩn hóa ảnh
    tf.keras.layers.Rescaling(1./255, input_shape=(100,100,3)),

    # CNN block 1
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # CNN block 2
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # CNN block 3
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # phân loại
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# =========================
# 4. COMPILE
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 5. TRAIN
# =========================

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

# =========================
# 6. EVALUATE
# =========================

loss, acc = model.evaluate(test_ds)

print("\nTEST ACCURACY:", acc)

# =========================
# 7. SAVE MODEL
# =========================

model.save("model.keras")

# =========================
# 8. VẼ BIỂU ĐỒ
# =========================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["train","test"])
plt.show()