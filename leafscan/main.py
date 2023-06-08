import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from data import download_data
from model import initialize_model, compile , train, evaluate

def operationnal() :
#comment data_dir first time you use this cell
    train_ds , val_ds , test_ds = download_data()

    model =initialize_model()
    compile(model)
    model, history = train(model, train_ds, val_ds)
    evaluate(model, test_ds)

    model.evaluate(test_ds)

    model.save('../models')
    print("✅ model saved")
operationnal()

def pred(X):
    model = keras.models.load_model('../models')
    y_pred = model.predict(X)
    return y_pred
