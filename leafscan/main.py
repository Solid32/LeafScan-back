import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import models, load_model
from tensorflow.keras.callbacks import EarlyStopping
from data import download_data
from model import initialize_model, compile , train, evaluate

def operationnal(retrain=False, epoch=20) :
#comment data_dir first time you use this cell
    train_ds , val_ds , test_ds = download_data()
    if retrain == False:
        model = initialize_model()
        print("🚨 model initialized")
    else :
        model = load_model('../models')
        print("🚨 model loaded")
    compile(model)
    model, history = train(model, train_ds, val_ds)
    evaluate(model, test_ds, epoch)
    print("✅ model evaluate")
    model.save('../models')
    print("✅ model saved")


def pred(X):
    model = load_model('../models')
    y_pred = model.predict(X)
    return y_pred

if __name__ == '__main__':
    operationnal()
