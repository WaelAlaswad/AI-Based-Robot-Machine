import keras
import os
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint  # to save the results in every iteration
import matplotlib.pyplot as plt
import cv2

# Assign the train and validation data
train = ImageDataGenerator(
    zoom_range=0.2, rotation_range=20
)  # shear_range : Prespective angle
train_batches = train.flow_from_directory(
    directory="/content/drive/MyDrive/AiandML/Camera2/Training/Face",
    target_size=(224, 224),
    classes=["Bad", "Good"],
    batch_size=50,
)

valid = ImageDataGenerator()  # shear_range : Prespective angle
valid_batches = train.flow_from_directory(
    directory="/content/drive/MyDrive/AiandML/Camera2/Training/Face",
    target_size=(224, 224),
    classes=["Bad", "Good"],
    batch_size=50,
)
print(len(train_batches))

# VGG16 Summary
vgg_16 = VGG16()
vgg_16.summary()


# Editing the VGG16 Layers
transferred_model = Sequential()

for layer in vgg_16.layers[:-3]:
    transferred_model.add(layer)

for layer in transferred_model.layers:
    layer.trainable = False

transferred_model.add(Dense(4096, activation="relu"))
transferred_model.add(Dense(4096, activation="relu"))
transferred_model.add(Dense(2, activation="sigmoid"))

# Model compiling
transferred_model.compile(
    optimizer=Adam(lr=0.01), loss=categorical_crossentropy, metrics=["accuracy"]
)
transferred_model.summary()


# Model fitting
checkpoint = ModelCheckpoint(
    "vgg16.h5", monitor="val_loss", save_best_only=True, mode="auto", period=1
)  # Period : Saving period
transferred_model.fit_generator(
    generator=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=100,
    callbacks=[checkpoint],
)

# Model saving
transferred_model.save("/content/drive/MyDrive/AiandML/TLVGG16FaceMask.h5")
