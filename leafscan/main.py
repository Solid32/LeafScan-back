import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd


#comment data_dir first time you use this cell
data_dir = '../raw_data'
checkpoint_filepath = '../models/tmp'

dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size=32,
    image_size=(256,256),
    shuffle=True,
    seed=42
    )

train_val_ds = dataset.take(round(len(dataset) * 0.9))  # 90% pour l'entraînement
test_ds = dataset.skip(round(len(dataset) * 0.9))  # 10% pour le test

train_ds = train_val_ds.take(round(len(train_val_ds) * 0.8))  # 80% pour l'entraînement
val_ds = train_val_ds.skip(round(len(train_val_ds) * 0.8))  # 20% pour le test


#Needed to be able to use 2 GPU's on google VM
#<--
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
#-->
    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(256,256,3)),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.CenterCrop(height=224, width=224),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.Conv2D(64, 4, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(246, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(39, activation='softmax')
    ])

initial_learning_rate = 0.0015
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(patience=4, restore_best_weights=True)


#The 'model_checkpoint_callback' save the result of each epoch on the disk
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


model.fit(train_ds,
          callbacks=[es,model_checkpoint_callback],
          validation_data=val_ds,
          epochs=1)

model.save('../models')

model.evaluate(test_ds)
