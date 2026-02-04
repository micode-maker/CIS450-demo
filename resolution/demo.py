# demo loading image

import cv2 as cv
import sys
import os

print("Current working directory:", os.getcwd())

img = cv.imread(cv.samples.findFile("../photos/1.jpg"))
print(img.shape)

if img is None:
    sys.exit("Could not read the image.")

cv.namedWindow("Display window", cv.WINDOW_NORMAL)
cv.resizeWindow("Display window", 600, 800)

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("../photos/1.jpg", img)

resized_image = cv.resize(img, (640, 853), dst=None, fx=0, fy=0, interpolation=cv.INTER_LINEAR)
filename = "../photos/IMG1-640x853.jpg"
cv.imwrite(filename, resized_image)
print(f"Image saved to {filename}")