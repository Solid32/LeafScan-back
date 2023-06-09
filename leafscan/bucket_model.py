from google.cloud import storage

BUCKET_NAME = "leafscan"

storage_filename = "models/models/saved_model.pb"
local_filename = "../models/saved_model.pb"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(storage_filename)
blob.download_to_filename(local_filename)
