
import tensorflow as tf

def download_data(path=None, batch_size=32, image_size=(256,256),shuffle=True, testratio=0.9, valratio=0.8, color_mode='rgb') :

    if path == None :
        data_dir = '../raw_data'
    else :
        data_dir = path

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        batch_size=batch_size,
        image_size=image_size,
        color_mode=color_mode,
        shuffle=shuffle,
        seed=42 )
    train_val_ds = dataset.take(round(len(dataset) * testratio))  # 90% pour l'entraînement
    test_ds = dataset.skip(round(len(dataset) * testratio))  # 10% pour le test

    train_ds = dataset.take(round(len(train_val_ds) * valratio))  # 90% pour l'entraînement
    val_ds = dataset.skip(round(len(train_val_ds) * valratio))  # 10% pour le test

    print("✅ data loaded")

    return train_ds , test_ds , val_ds
