# Final Project
## Description
For my final project, I will try to create an automated Face or Object Detection tool. The application will allow users to upload photos and automatically identify human faces or specific objects by drawing bounding boxes around them.

## Design
Containerised using Docker, my project builds on a Flask web application structure with OpenCV for image processing. The current template handles image uploads and edge detection, I will modify the detect_edges() function to use OpenCV's Haar Cascade classifiers for face detection. Or a DNN module with pre-trained models like YOLO for general object detection. The workflow will remains similar, user uploads an image via the web form, the backend processes it with cv2.CascadeClassifier.detectMultiScale() or cv2.dnn.readNet() draws bounding boxes using cv2.rectangle() and returns the annotated image. I'll store the cascade XML files or model weights in the project directory and load them at startup.

### Resources
OpenCV Face Detection Tutorial: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html

OpenCV DNN Object Detection: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html

Flask File Uploads: https://flask.palletsprojects.com/en/stable/patterns/fileuploads/

Haar Cascade Files (GitHub): https://github.com/opencv/opencv/tree/master/data/haarcascades

### AI Usage
I used Claude to help structure my prototyping approach and identify the relevant OpenCV functions (detectMultiScale, cv2.dnn) for transitioning from edge detection to face/object detection. I also asked for guidance on which pre-trained models are compatible with opencv-python-headless.
