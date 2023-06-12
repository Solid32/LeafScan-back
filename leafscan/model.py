import numpy as np
import os
import tensorflow as tf
# import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16

def initialize_model(shape):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
    #-->

        # Charger le modèle MobileNet sans les poids pré-entraînés
        gigi = VGG16(
            input_shape=shape,
            include_top=False,
            weights='imagenet'
        )
        gigi.trainable = False
        model = tf.keras.Sequential([
          tf.keras.layers.Rescaling(1./255, input_shape=(256,256,3)),
          tf.keras.layers.RandomFlip("horizontal_and_vertical"),
          tf.keras.layers.RandomRotation(0.2),
          gigi,
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dropout(0.15),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(39, activation='softmax')
        ])
    print("✅ Model created")
    # Afficher les informations sur le modèle
    return model

def compile(model, lr_rate = 0.002, dc_steps = 2000, dc_rate = 0.9):
    initial_learning_rate = lr_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=dc_steps,
    decay_rate=dc_rate
)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


    print("✅ Model compiled")

    return model

def train(model, train_ds , val_ds, epochs = 20, patience = 5):
    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained on {train_ds.cardinality()} rows with min val acc: {round(np.min(history.history['accuracy']), 2)}")

    return model, history

def evaluate(model, test_ds):

    value = model.evaluate(test_ds)
    return value
