import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Load data from the segmented signatures folder
def load_data(output_folder):
    X = []
    y = []
    
    # Loop through each group folder
    for group_folder in os.listdir(output_folder):
        group_path = os.path.join(output_folder, group_folder)
        
        # Check if the path is a directory
        if os.path.isdir(group_path):
            # Loop through each person's folder within the group
            for person_folder in os.listdir(group_path):
                person_path = os.path.join(group_path, person_folder)
                
                if os.path.isdir(person_path):
                    label = person_folder  # Use folder name as label
                    
                    # Loop through each image file in the person's folder
                    for image_file in os.listdir(person_path):
                        image_path = os.path.join(person_path, image_file)
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
                            X.append(image)
                            y.append(label)
                        else:
                            print(f"Image {image_file} could not be loaded.")
                else:
                    print(f"{person_folder} is not a directory.")
        else:
            print(f"{group_folder} is not a directory.")
    
    print(f"Loaded {len(X)} images and {len(y)} labels.")
    return np.array(X), np.array(y)

# Preprocess the data
def preprocess_data(X, y):
    X = X.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    y, unique_indices = pd.factorize(y)  # Convert labels to numeric (factorized)
    y = to_categorical(y, num_classes=len(unique_indices))  # One-hot encoding
    return X, y, len(unique_indices)

# Extract HOG features from a list of images
def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# Load and preprocess the data
X, y = load_data('Grouped_Output')  # Load signature images
X, y, NUM_CLASSES = preprocess_data(X, y)  # Preprocess images and labels

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels for SVM (convert one-hot encoded labels back to integers)
y_train_encoded = np.argmax(y_train, axis=1)  # Convert y_train from one-hot to integer labels
y_val_encoded = np.argmax(y_val, axis=1)      # Convert y_val from one-hot to integer labels

# Extract HOG features for both training and validation sets
X_train_hog = extract_hog_features(X_train)  # Extract HOG features for training set
X_val_hog = extract_hog_features(X_val)      # Extract HOG features for validation set

# Train SVM on HOG features
svm_model = SVC(kernel='linear')  # Use linear kernel for SVM
svm_model.fit(X_train_hog, y_train_encoded)  # Train SVM model

# Predict using the trained SVM model on validation set
y_pred_svm = svm_model.predict(X_val_hog)

# Evaluate the SVM model performance
svm_accuracy = accuracy_score(y_val_encoded, y_pred_svm)
svm_precision = precision_score(y_val_encoded, y_pred_svm, average='macro')
svm_recall = recall_score(y_val_encoded, y_pred_svm, average='macro')
svm_f1 = f1_score(y_val_encoded, y_pred_svm, average='macro')

# Print out the performance metrics
print(f'HOG+SVM Validation Accuracy: {svm_accuracy:.4f}')
print(f'HOG+SVM Precision: {svm_precision:.4f}')
print(f'HOG+SVM Recall: {svm_recall:.4f}')
print(f'HOG+SVM F1-Score: {svm_f1:.4f}')

# Plotting the evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [svm_accuracy, svm_precision, svm_recall, svm_f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1)  # All metrics are between 0 and 1
plt.title('HOG+SVM Evaluation Metrics')
plt.ylabel('Score')
plt.show()
