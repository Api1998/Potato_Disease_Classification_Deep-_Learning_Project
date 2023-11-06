# import libraries
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "api-tf-models"
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

model = None
interpreter = None
input_index = None
output_index = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# Predict function to be deployed in GCP
def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/potatoes.h5",
            "/tmp/potatoes.h5",
        )
        model = tf.keras.models.load_model("/tmp/potatoes.h5")

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256))  # image resizing
    )

    image = image / 255  # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}
