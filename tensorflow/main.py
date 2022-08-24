import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten





image_size = (160, 160)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../imagenette2-160/train",
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../imagenette2-160/val",
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
)

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


model = Sequential()
model.add(Input(shape=(*image_size, 3)))  # 250x250 RGB images
model.add(Conv2D(32, 5, activation="relu"))
model.add(Conv2D(64, 5, activation="relu"))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))

model.summary()

model.fit(train_ds, epochs=5, validation_data=val_ds)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
