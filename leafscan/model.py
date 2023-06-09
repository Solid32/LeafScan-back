import numpy as np
import os
import tensorflow as tf
# import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

def initialize_model(shape):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
    #-->
        model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=shape),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.CenterCrop(height=224, width=224),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.Conv2D(64, 4, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(246, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(39, activation='softmax')
        ])
    return model

def compile(model, lr_rate = 0.0015, dc_steps = 2000, dc_rate = 0.9):
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
