from google.cloud import storage
from leafscan.params import *



def verify_model_is_loaded():
    '''
    Check if the model exist localy.  If not, call the function load_from_bucket
    '''
    if os.path.isfile(MODELS_LOCATION + "/" + PROD_MODEL_NAME):
        print('model already exists on ' + MODELS_LOCATION + "/" + PROD_MODEL_NAME)
    else:
        print('copying  model from bucket ....')
        load_from_bucket()
        print('model copied !')

def load_from_bucket():
    '''
    Copy the model from Google cloud bucket to the local machie
    '''
    destination_file_name = MODELS_LOCATION +"/"+ PROD_MODEL_NAME
    source_blob_name = MODELS_LOCATION +"/"+ PROD_MODEL_NAME

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    return None
