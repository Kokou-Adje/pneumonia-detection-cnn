"""
Pneumonia detection from chest X-ray images with a convolutional neural network.
Author: Kokou Adje
"""
### Import libraries
import os
import tensorflow as tensrflw
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as matplot
import seaborn as snsb

# Defining the path for images directories
prjct_dirctry = os.path.dirname(__file__)
train_dirctry = os.path.join(prjct_dirctry, 'train')
var_dirctry = os.path.join(prjct_dirctry, 'val')
test_dirctry = os.path.join(prjct_dirctry, 'test')

# Setting of the Data Preprocessing
train_datagenerated_cnn = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
value_test_datagenerated_cnn = ImageDataGenerator(rescale=1.0 / 255)
generator_training = train_datagenerated_cnn.flow_from_directory(
    train_dirctry,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)
value_generator = value_test_datagenerated_cnn.flow_from_directory(
    var_dirctry,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)
value_generator_val = value_test_datagenerated_cnn.flow_from_directory(
    test_dirctry,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# CNN Architecture of the new model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# New Model compilation phase
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# New Model Training phase
early_stopping_value = tensrflw.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = cnn_model.fit(
    generator_training,
    validation_data=value_generator,
    epochs=50,
    callbacks=[early_stopping_value]
)
# New Model Performance Evaluation
test_loss_model, test_accuracy_model = cnn_model.evaluate(value_generator_val)
print(f"Test Accuracy: {test_accuracy_model:.2f}")

# Report Generating: Confusion Matrix
y_pred_model = cnn_model.predict(value_generator_val)
y_pred_model_classes = (y_pred_model > 0.5).astype('int').flatten()
y_true_model = value_generator_val.classes
print(classification_report(y_true_model, y_pred_model_classes, target_names=['Normal', 'Pneumonia']))
conf_matrix_model = confusion_matrix(y_true_model, y_pred_model_classes)
snsb.heatmap(conf_matrix_model, annot=True, cmap='Blues', fmt='g', xticklabels=['Normal', 'Pneumonia'],yticklabels=['Normal', 'Pneumonia'])
matplot.xlabel('Predicted')
matplot.ylabel('Actual')
matplot.title('Confusion Matrix')
matplot.show()

# Training History for the new model
matplot.figure(figsize=(12, 4))
matplot.subplot(1, 2, 1)
matplot.plot(history.history['accuracy'], label='Training Accuracy')
matplot.plot(history.history['val_accuracy'], label='Validation Accuracy')
matplot.xlabel('Epochs')
matplot.ylabel('Accuracy')
matplot.legend()
matplot.title('Training and Validation Accuracy')
matplot.subplot(1, 2, 2)
matplot.plot(history.history['loss'], label='Training Loss')
matplot.plot(history.history['val_loss'], label='Validation Loss')
matplot.xlabel('Epochs')
matplot.ylabel('Loss')
matplot.legend()
matplot.title('Training and Validation Loss')
matplot.show()

# Save the trained model so the Streamlit app (app.py) can load it for predictions
model_path = os.path.join(prjct_dirctry, 'pneumonia_cnn_model.keras')
cnn_model.save(model_path)
print(f"Model saved to {model_path}")
