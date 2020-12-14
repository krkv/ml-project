import pandas as pd
import numpy as np
import tensorflow as tensorflow

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50

import json

train2020 = pd.read_csv(
    "../input/siim-isic-melanoma-classification/train.csv", dtype=str)
test2020 = pd.read_csv(
    "../input/siim-isic-melanoma-classification/test.csv", dtype=str)
train2019 = pd.read_csv(
    "../input/isic-2019/ISIC_2019_Training_GroundTruth.csv", dtype=str)
bal2020 = pd.concat([neg2020[:2000], pos2020])


def append_jpg(fn):
    return fn+".jpg"


def append_png(fn):
    return fn+".png"


def add_target(mel):
    if(mel == "1.0"):
        return "1"
    else:
        return "0"


train2019["file_name"] = train2019["image"].apply(append_jpg)
train2019["target"] = train2019["MEL"].apply(add_target)

train2020["file_name"] = train2020["image_name"].apply(append_png)
test2020["file_name"] = test2020["image_name"].apply(append_png)

bal2020["file_name"] = bal2020["image_name"].apply(append_png)

train_datagen = ImageDataGenerator(
    rotation_range=90,
    shear_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    channel_shift_range=0.7,
    brightness_range=[0.7, 1.3],
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(
    rotation_range=90,
    shear_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    channel_shift_range=0.7,
    brightness_range=[0.7, 1.3],
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=bal2020,
    directory="../input/siic-isic-224x224-images/train",
    x_col="file_name",
    y_col="target",
    class_mode="binary",
    target_size=(224, 224),
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test2020,
    directory="../input/siic-isic-224x224-images/test",
    x_col="file_name",
    y_col=None,
    class_mode=None,
    target_size=(224, 224)
)

resnet = ResNet50(weights='imagenet', include_top=False,
                  input_shape=(224, 224, 3))

for layer in resnet.layers[:75]:
    layer.trainable = False

for layer in resnet.layers[75:]:
    layer.trainable = True

model = models.Sequential()
model.add(resnet)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(2048, activation='relu',
                       kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['AUC'])

print(model.summary())

model.fit_generator(train_generator, epochs=5)

predictions = model.predict_generator(test_generator)

submission = pd.DataFrame()

submission['image_name'] = test2020['image_name']
submission['target'] = predictions

submission.to_csv('submission.csv', index=False)
