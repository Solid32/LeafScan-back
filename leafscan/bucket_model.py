from google.cloud import storage
import os

PROD_MODEL_NAME = os.getenv('PROD_MODEL_NAME')
MODELS_LOCATION = os.getenv('MODELS_LOCATION')
BUCKET_NAME = os.getenv('BUCKET_NAME')

def verify_model_is_loaded():
    '''
    Check if the model  exist localy.  If not, call the function load_from_bucket
    '''


    if os.path.isfile(MODELS_LOCATION + PROD_MODEL_NAME):
        print('model already exists on /models')
    else:
        print('coping  model from bucket ....')
        load_from_bucket()
        print('model copied !')

def load_from_bucket():
    '''
    Copy the model from Google cloud bucket to the local machie
    '''

    destination_file_name = MODELS_LOCATION + PROD_MODEL_NAME
    source_blob_name = BUCKET_NAME + MODELS_LOCATION + PROD_MODEL_NAME

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    return None
