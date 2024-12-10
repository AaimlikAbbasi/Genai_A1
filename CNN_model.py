import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# Path to your image folder
data_path = "C:\\21i2540_GenAi_A1\\templates\\signated_images"

# Load images and labels
def load_images_and_labels(data_path):
    images = []
    labels = []
    label_map = {}
    label_counter = 0
    
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".png") or file_name.endswith(".jpg"):
                    img_path = os.path.join(folder_path, file_name)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (128, 128))  # Resize to 128x128
                    images.append(image)
                    
                    # Assign labels based on folder names
                    if folder not in label_map:
                        label_map[folder] = label_counter
                        label_counter += 1
                    labels.append(label_map[folder])
    
    images = np.array(images).reshape(-1, 128, 128, 1) / 255.0  # Normalize
    labels = np.array(labels)
    return images, labels, label_map

# Load data
images, labels, label_map = load_images_and_labels(data_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# CNN Model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_map), activation='softmax')  # Output layer with number of classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# CNN training
cnn_model = create_cnn_model()
history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# CNN evaluation
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f'CNN Test Accuracy: {cnn_accuracy:.4f}')

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features = hog(img.reshape(128, 128), pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)  # Removed `multichannel`
        hog_features.append(features)
    return np.array(hog_features)

# HOG feature extraction
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Train SVM on HOG features
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_hog, y_train)

# SVM evaluation
y_pred_svm = svm_model.predict(X_test_hog)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='macro')
svm_recall = recall_score(y_test, y_pred_svm, average='macro')
svm_f1 = f1_score(y_test, y_pred_svm, average='macro')

print(f'HOG+SVM Test Accuracy: {svm_accuracy:.4f}')
print(f'HOG+SVM Precision: {svm_precision:.4f}')
print(f'HOG+SVM Recall: {svm_recall:.4f}')
print(f'HOG+SVM F1-Score: {svm_f1:.4f}')

# CNN evaluation metrics
y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
cnn_precision = precision_score(y_test, y_pred_cnn, average='macro')
cnn_recall = recall_score(y_test, y_pred_cnn, average='macro')
cnn_f1 = f1_score(y_test, y_pred_cnn, average='macro')

print(f'CNN Precision: {cnn_precision:.4f}')
print(f'CNN Recall: {cnn_recall:.4f}')
print(f'CNN F1-Score: {cnn_f1:.4f}')

# Full classification report for CNN
print("\nCNN Classification Report:")
print(classification_report(y_test, y_pred_cnn))

# Plot accuracy comparison
models = ['CNN', 'HOG+SVM']
accuracies = [cnn_accuracy, svm_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: CNN vs HOG+SVM')
plt.ylim(0, 1)  # Set y-axis limits to show percentages clearly
plt.show()

# Confusion matrices
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_map))
    plt.xticks(tick_marks, label_map.keys(), rotation=45)
    plt.yticks(tick_marks, label_map.keys())
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(y_test, y_pred_cnn, 'CNN Confusion Matrix')
plot_confusion_matrix(y_test, y_pred_svm, 'HOG+SVM Confusion Matrix')


# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()