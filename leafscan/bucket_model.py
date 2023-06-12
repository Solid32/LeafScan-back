from google.cloud import storage
import os

def verify_model_is_loaded():
    '''
    Check if the model  exist localy.  If not, call the function load_from_bucket
    '''
    if os.path.isfile("models/" + PROD_MODEL_NAME):
        print('model already exists on /models')
    else:
        print('coping  model from bucket ....')
        load_from_bucket()
        print('model copied !')

def load_from_bucket():
    '''
    Copy the model from Google cloud bucket to the local machie
    '''
    BUCKET_NAME = "leafscan"
    storage_filename = "models/models/"
    local_filename = "/home/rdurs/code/solid32/LeafScan-back/models/"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=storage_filename)

    for blob in blobs:
        if "/" in blob.name:
            # Extract the filename from the full path
            filename = blob.name.split("/")[-1]
            print(filename)
        else:
            filename = blob.name
            print(filename)

        blob.download_to_filename(local_filename + filename)

    return None
