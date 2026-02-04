
import cv2 as cv
import sys
import os

print("Current working directory:", os.getcwd())

photos_dir = "../photos"

for filename in os.listdir(photos_dir):
    img_path = os.path.join(photos_dir, filename)
    img = cv.imread(img_path)
    
    if img is None:
        sys.exit("Could not read the image.")
    
    resized_image = cv.resize(img, (640, 853), dst=None, fx=0, fy=0, interpolation=cv.INTER_LINEAR)
    
    name, ext = os.path.splitext(filename)
    filename = f"../resolution/{name}-640x853.jpg"
    
    cv.imwrite(filename, resized_image)
    print(f"Image saved to {filename}")