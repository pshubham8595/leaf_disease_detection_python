import PIL
import keras.models
import keras_preprocessing.image
import numpy
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Sample data
details = {
    'name': 'John Doe',
    'age': 30,
    'email': 'john.doe@example.com'
}

print("--------------------------")
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
CHANNELS = 3
NUM_CLASSES = 3  # Number of classes (healthy, disease1, disease2)
TRAIN_DATA_DIR = 'dataset_leafs/Train'
VALIDATION_DATA_DIR = 'dataset_leafs/Validation'
MODELS_DIR = 'models'

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset_leafs",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)
print(dataset)
class_names = dataset.class_names
print(class_names)
print("--------------------------")


def get_data_set_partition(ds, train_split=0.1, test_split=0.1, val_split=0.8, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        train_dataset = ds.take(train_size)
        val_dataset = ds.skip(train_size).take(val_size)
        test_dataset = ds.skip(train_size).skip(val_size)
        print("TrainingDataSet Length: ", len(train_dataset))
        print("ValidationDataSet Length: ", len(val_dataset))
        print("TestDataSet Length: ", len(test_dataset))
        return train_dataset, val_dataset, test_dataset


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    return image_array


def predict_leaf_disease(prediction_model, img):
    prediction_made = prediction_model.predict(img)
    predicted_class = class_names[np.argmax(prediction_made[0])]
    return predicted_class


def predict_image(image_path):
    train_ds, val_ds, test_ds = get_data_set_partition(dataset)
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    model = tf.keras.models.load_model('high_accuracy_model.h5')
    # test_image_path = 'dataset_leafs/Potato___Late_blight/0b092cda-db8c-489d-8c46-23ac3835310d___RS_LB 4480.JPG'

    image_path = preprocess_image(image_path)

    predicted_class_name = predict_leaf_disease(model, image_path)
    print(predicted_class_name)

    return predicted_class_name


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['image']

        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            # Create the uploads directory if it doesn't exist
            if not os.path.exists('cache'):
                os.makedirs('cache')

            # Save the uploaded image

            image_path = os.path.join('cache', file.filename)
            full_image_path = os.path.abspath(image_path)
            file.save(image_path)
            print("Selected Image Path: ", full_image_path)

            # Perform prediction
            prediction = predict_image(image_path)

            # Remove the uploaded image
            os.remove(image_path)

            # Pass prediction result and image path to HTML
            return render_template('index.html', prediction=prediction, image=full_image_path)

    # Render the HTML page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

# --- MODEL PREPARATION ----
#
# resize_and_rescale = tf.keras.Sequential([
#     layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
#     layers.experimental.preprocessing.Rescaling(1.0/255)
# ])
#
# data_augmentation = tf.keras.Sequential([
#
#     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#     layers.experimental.preprocessing.RandomRotation(0.2)
# ])
# input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
# model = models.Sequential([
#     resize_and_rescale,
#     data_augmentation,
#     layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
#     layers.MaxPool2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#     layers.MaxPool2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#     layers.MaxPool2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation="relu"),
#     layers.MaxPool2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation="relu"),
#     layers.MaxPool2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation="relu"),
#     layers.MaxPool2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation="relu"),
#     layers.Dense(NUM_CLASSES, activation="softmax")
# ])
# model.build(input_shape=input_shape)
# print(model.summary())

# model.compile(
#     optimizer="adam",
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=["accuracy"]
# )
# history = model.fit(
#     train_ds,
#     batch_size=BATCH_SIZE,
#     validation_data=val_ds,
#     verbose=1,
#     epochs=EPOCHS
# )
# print(history)
# print(os.getcwd())
# model.save("high_accuracy_model.h5")

# ---------------------------------------------------
# print(model.summary())
# scores = model.evaluate(test_ds)
# print(scores)

# Perfect Healthy : test_image_path = 'dataset_leafs/Potato___healthy/0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG'
# Perfect Healthy : test_image_path = 'dataset_leafs/Potato___Early_blight/1af20ff8-980d-4912-b337-804b09667de3___RS_Early.B 7392.JPG'


#
# for images_batch, labels_batch in test_ds.take(1):
#     first_image = images_batch[0].numpy().astype('uint8')
#     print("First Image to predict")
#     batch_prediction = model.predict(images_batch)
#     print("Actual Label: ", class_names[labels_batch[0].numpy()])
#     print("Predicted Label: ", class_names[np.argmax(batch_prediction[0])])
#
#     plt.imshow(first_image)
#     plt.show()

# Preprocess the test image


# img_pre_processed = pre_process_image(test_image_path)
# predictions = model.predict(img_pre_processed)
# print(predictions[0])
#
# first_className = class_names[0]
# second_className = class_names[1]
# third_className = class_names[2]
#
# print(first_className, second_className, third_className)
#
# predicted_class_name = class_names[np.argmax((predictions[0]))]
# print("Prediction: ", predicted_class_name)

# def predict_image_disease(saved_model, img):
#     img_pre_processed = pre_process_image(img)
#     predictions = saved_model.predict(img_pre_processed)
#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * np.max(predictions[0], 2))
#     return predicted_class, confidence
#
#
# predicted_disease_image_class, disease_image_confidence = predict_image_disease(model, test_image_path)
# print(predicted_disease_image_class)
# print(disease_image_confidence)
