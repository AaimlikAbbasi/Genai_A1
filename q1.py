

import os
import numpy as np
import cv2

# Path to the directory containing images
img_dir = "C:\\21i2540_GenAi_A1\\Data"
save_dir = "C:\\21i2540_GenAi_A1\\templates\\signated_images"
os.makedirs(save_dir, exist_ok=True)

# Get list of all image files in the directory
image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
id_num = 0

# Process each image in the directory
for img_idx, image_file in enumerate(image_files):
    # Construct full image path
    img_path = os.path.join(img_dir, image_file)

    # Read the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Removing Noise using a Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Thresholding image to create a binary image
    _, thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY_INV)
    
    # Finding Contours for grid structure
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = list()

    # Filter and sort the contours by area (to exclude large grid structure)
    mean_area = 0
    number = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 900000 > area > 100000:  # Filter based on area (adjust as needed)
            mean_area += area
            number += 1

    mean_area /= number
    mean_area = int(mean_area)

    # Filter and sort the contours by area (to exclude large grid structure)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 900000 > area > mean_area - (mean_area * .4):  # Filter based on area (adjust as needed)
            filtered_contours.append((x, y, w, h, contour))

    # Sort contours row-wise (up to down) and right to left
    filtered_contours = sorted(filtered_contours, key=lambda c: (c[1], -c[0]))

    # Group contours by rows
    row_height_threshold = 100  # Adjust based on the vertical spacing between rows
    current_row = []
    rows = []

    for i, (x, y, w, h, contour) in enumerate(filtered_contours):
        if not current_row:
            current_row.append((x, y, w, h, contour))
        else:
            # If the y-coordinate difference is larger than the row height threshold, start a new row
            if abs(current_row[-1][1] - y) > row_height_threshold:
                rows.append(current_row)
                current_row = [(x, y, w, h, contour)]
            else:
                current_row.append((x, y, w, h, contour))

    # Append the last row
    if current_row:
        rows.append(current_row)

    # Save each signature row-wise from right to left
    for row_idx, row in enumerate(rows):
        # Sort each row by the x-coordinate (right to left)
        row = sorted(row, key=lambda c: -c[0])  # Sort right to left

        # Create a unique folder for this image (e.g., id_1, id_2, ...)
        image_save_dir = os.path.join(save_dir, f"{id_num + 1}")
        os.makedirs(image_save_dir, exist_ok=True)
        id_num += 1

        img_num = 1
        for col_idx, (x, y, w, h, contour) in enumerate(row):
            # Extract the signature ROI (Region of Interest)
            signature_roi = image[y:y + h, x:x + w]
            
            # Save the signature as an image in the corresponding folder
            signature_filename = f"{img_num}.jpg"
            signature_save_path = os.path.join(image_save_dir, signature_filename)
            cv2.imwrite(signature_save_path, signature_roi)
            img_num += 1

    print(f"Processed and saved signatures for image: {image_file} in folder {image_save_dir}")

print("Finished processing all images.")
