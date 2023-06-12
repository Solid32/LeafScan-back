from google.cloud import storage

def load_from_bucket():
    breakpoint()
    BUCKET_NAME = "leafscan"
    storage_filename = "models/models/"
    local_filename = "models/"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=storage_filename)

    for blob in blobs:
        if "tmp" not in blob.name:
            # Extract the filename from the full path
            filename = blob.name.split("/")[-1]
        # else:
        #     filename = blob.name

            blob.download_to_filename(local_filename + filename)

    return None


if __name__ == "__main__":
    load_from_bucket()
