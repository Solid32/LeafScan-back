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

        i = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
        x = tf.cast(i, tf.float32)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        core = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

        # Geler les poids du noyau
        core.trainable = False

        x = core(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(39, activation='softmax')(x)
        model = tf.keras.Model(inputs=[i], outputs=[x])

    print('************************************************************')
    print("✅ Model created")
    print('************************************************************')
    return model

def compile(model, lr_rate = 0.001, dc_steps = 1000, dc_rate = 0.9):
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

    print('************************************************************')
    print("✅ Model compiled")
    print('************************************************************')

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
    print('************************************************************')
    print(f"✅ Model trained on {train_ds.cardinality()} rows with min val acc: {round(np.min(history.history['accuracy']), 2)}")
    print('************************************************************')

    return model, history

def evaluate(model, test_ds):

    value = model.evaluate(test_ds)
    return value
