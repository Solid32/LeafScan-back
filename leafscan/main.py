import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from data import download_data
from model import initialize_model, compile , train, evaluate

def operationnal(retrain=False, epoch=10) :

    train_ds , test_ds , val_ds = download_data()
    temp = train_ds.element_spec[0].shape


    shape = tuple(temp.as_list()[1:])

    if retrain == False:
        model = initialize_model(shape)
        print("🚨 model initialized")
    else :
        model = load_model('models')
        print("🚨 model loaded")
    compile(model)
    model, history = train(model, train_ds, val_ds)
    evaluate(model, test_ds)
    print("✅ model evaluate")
    model.save('../models')
    print("✅ model saved")


def pred(X):
    model = load_model('models')
    y_pred = model.predict(X)
    #The return needs to be changed to be 'stringify' to the fastAPI
    return y_pred

if __name__ == '__main__':
    operationnal(retrain=True, epoch=20)
    #pred(X)
