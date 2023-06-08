import tensorflow as tf
from model import initialize_model
from data import download_data

# Define your training parameters
learning_rate = 0.001
batch_size = 256
patience = 4
split_ratio = 0.1
data_dir = '../raw_data'
# Create the dataset from your TensorFlow dataset
dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        batch_size=batch_size,
        image_size=(256,256),
        shuffle=True,
        seed=42 )

# Calculate the length of the datasets
dataset_length = tf.data.experimental.cardinality(dataset).numpy()

# Calculate the number of chunks based on the split ratio and batch size
num_chunks = int(dataset_length / (batch_size * (1 - split_ratio)))

# Split the train dataset into chunks
train_chunks = tf.data.Dataset.range(num_chunks).interleave(lambda x: dataset.skip(x * batch_size).take(batch_size), num_parallel_calls=tf.data.AUTOTUNE)

# Define your model and optimizer
model = initialize_model()       # Your model here
optimizer = 'adam'  # Your optimizer here

# Define the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Define metrics for evaluation
metrics = ['accuracy']  # List of metrics you want to track during training

# Define variables for early stopping
best_loss = float('inf')
patience_count = 0

# Iterate over the chunks for training
for chunk_id, chunk in enumerate(train_chunks):
    print(f"Training on chunk {chunk_id}")

    # Split the chunk into training and validation sets
    train_length = int(len(chunk) * (1 - split_ratio))
    chunk_train = chunk.take(int(train_length * (1 - split_ratio)))
    chunk_val = chunk.skip(int(train_length * (1 - split_ratio)))

    # Prepare the training data
    X_train_chunk, y_train_chunk = zip(*list(chunk_train))

    # Prepare the validation data
    X_val_chunk, y_val_chunk = zip(*list(chunk_val))

    # Create a new model for each chunk (optional)
    model_chunk = tf.keras.models.clone_model(model)

    # Compile the model
    model_chunk.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # Train the model on the current chunk
    history = model_chunk.fit(
        tf.data.Dataset.from_tensor_slices((X_train_chunk, y_train_chunk)).batch(batch_size),
        epochs=1,
        validation_data=(tf.data.Dataset.from_tensor_slices((X_val_chunk, y_val_chunk)).batch(batch_size))
    )

    # Track the best loss for early stopping
    if history.history['val_loss'][0] < best_loss:
        best_loss = history.history['val_loss'][0]
        patience_count = 0
    else:
        patience_count += 1

    # Check if early stopping criteria are met
    if patience_count >= patience:
        print("Early stopping: No improvement in validation loss.")
        break
