
import tensorflow as tf
from params import *
from leafscan.model import *
from leafscan.data import download_data
from tensorflow.keras.models import load_model

train_ds , test_ds , val_ds = download_data()
temp = train_ds.element_spec[0].shape


shape = tuple(temp.as_list()[1:])

model = initialize_model(shape)
print("...trying to load the model on ")
model = load_model('/home/rdurs/code/solid32/LeafScan-back/models/models')
print("!!!Model loaded !!!!")
model.save(MODELS_LOCATION + "/" + PROD_MODEL_NAME)
