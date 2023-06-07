import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd

#comment data_dir first time you use this cell
data_dir = '/home/rdurs/LeafScan-back/raw_data'

train_val_ds, test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    batch_size=32,  # Ajustez la taille du lot selon vos besoins
    image_size=(256, 256),  # Ajustez la taille des images selon vos besoins
    shuffle=True,  # Mélange les données si nécessaire
    seed=42,  # Définit une graine pour la reproductibilité du mélange
    validation_split=0.1,  # Spécifie la proportion de données à utiliser pour la validation
    subset="both"  # Spécifie le sous-ensemble à charger (ensemble d'entraînement)
)
train_ds = train_val_ds.take(round(len(train_val_ds) * 0.8))  # 80% pour l'entraînement
val_ds = train_val_ds.skip(round(len(train_val_ds) * 0.8))  # 20% pour le test

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(256,256,3)),
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(38, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_ds,
    validation_data=val_ds,
    epochs=5)

model.evaluate(test_ds)


# if __name__ == '__main__':
#     #preprocess()
#     train()
#     evaluate()
#     pred()
