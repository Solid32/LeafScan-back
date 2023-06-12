import numpy as np
import matplotlib.pyplot as plt
import os
# import PIL
# import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from leafscan.data import download_data
from leafscan.bucket_model import load_from_bucket, verify_model_is_loaded
from leafscan.model import initialize_model, compile , train, evaluate
import json

def operationnal(retrain=False, epoch=25,color_mode='rgb') :

    train_ds , test_ds , val_ds = download_data(color_mode=color_mode)
    temp = train_ds.element_spec[0].shape


    shape = tuple(temp.as_list()[1:])

    if retrain == False:
        model = initialize_model(shape)
        print("🚨 model initialized")
    else :
        load_from_bucket()
        model = load_model('models')
        print("🚨 model loaded")
    compile(model)
    model, history = train(model, train_ds, val_ds, epoch)
    evaluate(model, test_ds)
    print("✅ model evaluate")
    model.save('../models/my_h5_model.h5')
    print("✅ model saved")
    train_loss = history.history['loss']

    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        has_val_loss = True
    else:
        has_val_loss = False

    # Set up the plot
    plt.plot(train_loss, color='blue',  label='Training Loss')

    # If validation loss is available, plot it as well
    if has_val_loss:
        plt.plot(val_loss,color='orange', label='Validation Loss')

    # Customize the plot
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to a file
    plt.savefig('learning_curve.png')

def translate(y_pred):
    dict_list = {0: 'Apple Apple scab',
                 1: 'Apple Black rot',
                 2: 'Apple Cedar apple rust',
                 3: 'Apple healthy',
                 4: 'Background without leaves',
                 5: 'Blueberry healthy',
                 6: 'Cherry Powdery mildew',
                 7: 'Cherry healthy',
                 8: 'Corn Cercospora leaf spot Gray leaf spot',
                 9: 'Corn Common rust',
                 10: 'Corn Northern Leaf Blight',
                 11: 'Corn healthy',
                 12: 'Grape Black rot',
                 13: 'Grape Esca (Black Measles)',
                 14: 'Grape Leaf blight (Isariopsis Leaf Spot)',
                 15: 'Grape healthy',
                 16: 'Orange Haunglongbing (Citrus greening)',
                 17: 'Peach Bacterial spot',
                 18: 'Peach healthy',
                 19: 'Pepper bell Bacterial spot',
                 20: 'Pepper bell healthy',
                 21: 'Potato Early blight',
                 22: 'Potato Late blight',
                 23: 'Potato healthy',
                 24: 'Raspberry healthy',
                 25: 'Soybean healthy',
                 26: 'Squash Powdery mildew',
                 27: 'Strawberry Leaf scorch',
                 28: 'Strawberry healthy',
                 29: 'Tomato Bacterial spot',
                 30: 'Tomato Early blight',
                 31: 'Tomato Late blight',
                 32: 'Tomato Leaf Mold',
                 33: 'Tomato Septoria leaf spot',
                 34: 'Tomato Spider mites Two-spotted spider mite',
                 35: 'Tomato Target Spot',
                 36: 'Tomato Tomato Yellow Leaf Curl Virus',
                 37: 'Tomato Tomato mosaic virus',
                 38: 'Tomato healthy'}
    cat_plant = np.argsort(y_pred[0])[::-1][:3]
    cat_test = y_pred[0][cat_plant]
    results = [dict_list[key] for key in cat_plant.tolist()]
    return {key:value for key, value in zip([dict_list[key] for key in cat_plant.tolist()],[[cat_test[i],cat_plant[i]] for i, elem in enumerate(results)])}

def pred(X):

    verify_model_is_loaded()

    model = load_model('models/prod_model.h5')
    y_pred = model.predict(X)
    dict_pred = translate(y_pred)

    return str(dict_pred)

if __name__ == '__main__':
    operationnal(retrain=False, epoch=50)
    #pred(X)
