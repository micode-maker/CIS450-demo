import cv2
import numpy as np
import os

# Path to image
input_path = "coins/coins.png"
output_path = "coins/coins_annotated.png"

# Load image
image = cv2.imread(input_path)
output = image.copy()

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Bronze / copper color range (adjust if needed)
lower_bronze = np.array([5, 85, 35])
upper_bronze = np.array([20, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_bronze, upper_bronze)

# Clean up mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

penny_count = 0

for contour in contours:
    area = cv2.contourArea(contour)
    
    # Filter small noise
    if area > 500:   # Adjust if necessary
        penny_count += 1
        
        # Draw contour
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

print("Number of pennies detected:", penny_count)

# Save annotated image
cv2.imwrite(output_path, output)