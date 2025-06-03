import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import warnings

warnings.filterwarnings("ignore")
not_fractured_dir = 'train/notfractured'
yes_fractured_dir = 'train/fractured'


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):  # List all files in the folder
        img = cv2.imread(f"{folder}/{filename}")  # Construct the path manually
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize
            images.append(img)
    return images

not_fractured_images = load_images_from_folder(not_fractured_dir)
yes_fractured_images = load_images_from_folder(yes_fractured_dir)

print(f"Loaded {len(not_fractured_images)} images from 'no' folder.")
print(f"Loaded {len(yes_fractured_images)} images from 'yes' folder.")

not_fractured_labels = [0] * len(not_fractured_images)
yes_fractured_labels = [1] * len(yes_fractured_images)

X = np.array(yes_fractured_images + not_fractured_images)
y = np.array(yes_fractured_labels + not_fractured_labels)


# Count the number of samples in each class
num_not_fractured = len(not_fractured_images)
num_yes_fractured = len(yes_fractured_images)

# Create the labels and values for the bar chart
labels = ['Not Fractured', 'Fractured']
values = [num_not_fractured, num_yes_fractured]

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['blue', 'purple'])
plt.title("Class Distribution: Not Fractured vs Fractured")
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.show()

print("Initial shape or dimensions of X", str(X.shape))
print ("Number of samples in our data: " + str(len(X)))
print ("Number of labels in our data: " + str(len(y)))
print("\n")
print ("Dimensions of images:" + str(X[0].shape))

figure = plt.figure()
plt.figure(figsize=(16, 10))

num_of_images = 50
classes = ["yes", "no"]

# Make sure X and y have enough data
if len(X) >= num_of_images and len(y) >= num_of_images:
    for index in range(0, num_of_images):  # Index 0 to num_of_images-1
        class_name = classes[y[index]]  # Get the class label
        plt.subplot(5, 10, index + 1)  # Adjust indexing for matplotlib (starts at 1)
        plt.imshow(X[index])  # Show the image (no need for cmap='gray_r' for color images)
        plt.title(f'{class_name}')
        plt.axis('off')
    plt.show()
else:
    print("Not enough images or labels!")

X = X.astype('float32')

img_rows = X[0].shape[0]
img_cols = X[0].shape[1]

input_shape = (img_rows, img_cols, 3)

X /= 255.0
y = to_categorical(y, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

L2 = 0.001

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.01), # 'adam'
              metrics = ['accuracy'])

print(model.summary())

# early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore weights from the best epoch
)

history = model.fit(X_train, y_train, batch_size=32,
                              epochs = 10,
                              validation_data = (X_test, y_test),
                              callbacks=[early_stopping],
                              verbose = 1,)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:\n", classification_report(y_true, y_pred_classes))
model.evaluate(X_test,y_test)

#accuracy curve
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()

#loss curve
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()


# Prediction function inside a loop
while True:
    input_image_path = input('Enter the path of the image: ')

    # Load the image
    input_image = cv2.imread(input_image_path)

    if input_image is None:
        print("Error: Could not read the input image. Please check the file path.")
        continue  # Ask the user for a new image path

    # Convert to RGB for display
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Display the original image
    plt.imshow(input_image_rgb)
    plt.axis('off')  # Turn off axis for cleaner display
    plt.show()

    # Resize and normalize the image
    try:
        input_image_reshape = cv2.resize(input_image, (224, 224))
        image_normalized = input_image_reshape / 255.0
    except Exception as e:
        print(f"Error during image resizing or normalization: {e}")
        continue  # Ask the user for a new image if error occurs

    # Reshape for model prediction
    img_reshape = np.reshape(image_normalized, (1, 224, 224, 3))
    img_reshape = img_reshape.astype('float32')  # Ensure correct data type

    # Make Predictions
    input_prediction = model.predict(img_reshape)

    # Display the prediction probabilities
    print('Prediction Probabilities are:', input_prediction)

    # Get the Predicted Label
    input_pred_label = np.argmax(input_prediction)

    # Interpret the result
    if input_pred_label == 1:
        print('Fractured')
    else:
        print('Not fractured')

    # Ask the user if they want to continue
    continue_check = input('Do you want to check another image? (yes/no): ').lower()
    if continue_check != 'yes':
        print("Exiting the program, Goodbye!")
        break  # Exit the loop if the user doesn't want to continue