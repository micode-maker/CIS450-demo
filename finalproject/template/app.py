"""
=============================================================================
File:        app.py
Project:     Face Detection Web Application
Author:      Michael
Course:      CIS 450 
Date:        April 2025
Description: 
    A Flask-based web application that allows users to upload images and 
    automatically detect human faces using YuNet, a deep learning face detector. 
    YuNet handles frontal, tilted, rotated, and profile faces in a single pass 
    with high accuracy.
    
    Detected faces are highlighted with styled bounding boxes and confidence
    scores.
 
Technical Stack:
    - Backend: Flask (Python web framework)
    - Face Detection: YuNet ONNX model via OpenCV DNN
    - Containerisation: Docker
    - Frontend: HTML/CSS with Jinja2 templating
 
Build Instructions:
    docker build -t app .
    docker run -p 80:80 app
    
Access:
    http://localhost
=============================================================================
"""
 
# =============================================================================
# IMPORTS
# =============================================================================
from flask import Flask, request, render_template_string, send_from_directory
import os
import cv2
import socket
import urllib.request
import time
 
# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================
app = Flask(__name__)
 
# Get the hostname for display
hostname = socket.gethostname()
 
# Directory for storing uploaded and processed images
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
# Directory for storing the downloaded model file
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
 
# =============================================================================
# MODEL URL AND PATH
# =============================================================================
# YuNet - Modern deep learning face detector from OpenCV's official model zoo
# Handles frontal, tilted, rotated, and profile faces in a single pass
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = os.path.join(MODEL_FOLDER, "face_detection_yunet.onnx")
 
 
# =============================================================================
# MODEL DOWNLOAD AND LOADING
# =============================================================================
 
def download_model():
    """
    Download the YuNet model file if it doesn't exist locally.
    
    Checks the models directory for the ONNX file. If not present,
    downloads it from OpenCV's official model zoo on GitHub.
    """
    # Skip if file already exists
    if os.path.exists(YUNET_PATH):
        return True
    
    # Download the model
    print("Downloading YuNet face detector")
    try:
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        print("YuNet downloaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to download YuNet: {e}")
        return False
 
 
def load_face_detector():
    """
    Load the YuNet face detector model.
    
    YuNet is a lightweight, state-of-the-art deep learning face detector
    that significantly outperforms traditional Haar Cascades. It handles
    frontal, tilted, rotated, and profile faces in a single pass.
    """
    # Ensure model file is available
    if not download_model():
        return None
    
    # Create the detector using OpenCV's FaceDetectorYN API
    try:
        detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,       # Path to the ONNX model file
            "",               # Config file (not needed for ONNX format)
            (320, 320),       # Initial input size (resized per image)
            0.6,              # Score threshold (confidence cutoff)
            0.3,              # NMS threshold (overlap cutoff)
            5000              # Top K (max detections before NMS)
        )
        print("YuNet face detector loaded successfully.")
        return detector
    except Exception as e:
        print(f"Failed to load YuNet: {e}")
        return None
 
 
# Load the face detector at application startup
face_detector = load_face_detector()
 
 
# =============================================================================
# FACE DETECTION
# =============================================================================
 
def detect_faces(input_path, output_path):
    """
    Algorithm Steps:
        1. Load the input image
        2. Configure YuNet with the image dimensions (required by the API)
        3. Run detection to get face bounding boxes and confidence scores
        4. Draw styled bounding boxes with corner accents for each face
        5. Add confidence label above each detected face
        6. Save the processed image and return the count
    
    YuNet Output Format:
        Each detected face is a row with 15 values:
        - [0:4] = bounding box (x, y, width, height)
        - [4:14] = 5 facial landmarks (eyes, nose, mouth corners)
        - [14] = confidence score (0.0 to 1.0)
    """
    # Check that the detector is available
    if face_detector is None:
        print("Face detector not available")
        return 0
    
    # Load the image
    img = cv2.imread(input_path)
    height, width = img.shape[:2]
    
    # Configure the detector for this image's dimensions
    # YuNet must know the input size to decode detections into correct coordinates
    face_detector.setInputSize((width, height))
    
    # Run detection
    # Returns: (retval, faces) where faces is an Nx15 numpy array
    retval, faces = face_detector.detect(img)
    
    # Handle case where no faces were detected
    if faces is None:
        cv2.imwrite(output_path, img)
        return 0
    
    # Draw bounding boxes for each detected face
    for face in faces:
        # Extract bounding box coordinates
        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        
        # Extract confidence score (last element)
        confidence = face[14]
        
        # Draw main rectangle in cyan-green
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 128), 2)
        
        # Draw corner accents for a modern look
        corner_len = min(w, h) // 4
        color = (0, 255, 200)  # Bright cyan-green
        thickness = 3
        
        # Top-left corner
        cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(img, (x, y), (x, y + corner_len), color, thickness)
        # Top-right corner
        cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)
        # Bottom-left corner
        cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)
        # Bottom-right corner
        cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)
        
        # Add confidence label above the face
        label = f"Face: {confidence*100:.0f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Position label above face, or below if near the top edge
        label_y = y - 10 if y - 10 > 10 else y + 20
        
        # Draw filled background rectangle for label
        cv2.rectangle(
            img,
            (x, label_y - label_size[1] - 5),
            (x + label_size[0] + 5, label_y + 5),
            color,
            -1  # Filled rectangle
        )
        # Draw label text in black for contrast
        cv2.putText(
            img, label, (x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )    
    
    # Save the processed image
    cv2.imwrite(output_path, img)
    
    return len(faces)
 
 
# =============================================================================
# HTML TEMPLATE
# =============================================================================
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e8e8e8;
        }
        .container { max-width: 960px; margin: 0 auto; padding: 40px 20px; }
        header { text-align: center; margin-bottom: 40px; }
        h1 {
            font-size: 2.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle { color: #888; font-size: 0.9rem; margin-top: 8px; }
        .card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
        }
        .upload-zone {
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 48px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-zone:hover { border-color: #00d9ff; background: rgba(0,217,255,0.05); }
        .upload-text { color: #aaa; margin-bottom: 20px; }
        input[type="file"] { display: none; }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            color: #1a1a2e;
            padding: 14px 32px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            font-size: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,217,255,0.4);
        }
        .btn-full { width: 100%; margin-top: 24px; font-size: 1.1rem; padding: 16px; }
        .file-name { margin-top: 16px; color: #00ff88; font-size: 0.9rem; }
        .stats {
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }
        .stat-box {
            flex: 1;
            min-width: 120px;
            background: rgba(0,217,255,0.1);
            border: 1px solid rgba(0,217,255,0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label { color: #888; font-size: 0.85rem; margin-top: 4px; }
        .images {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
        }
        .image-box h3 {
            font-size: 0.9rem;
            color: #888;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .image-box img {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .host { color: #555; font-size: 0.75rem; text-align: center; margin-top: 32px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Face Detection</h1>
            <p class="subtitle">Powered by YuNet deep learning model</p>
        </header>
 
        <div class="card">
            <form method="POST" action="/detect" enctype="multipart/form-data">
                <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                    <p class="upload-text">Click to select an image to detect faces</p>
                    <label class="btn">Choose File</label>
                    <input type="file" id="fileInput" name="file" accept="image/*" onchange="showFileName(this)">
                    <p class="file-name" id="fileName"></p>
                </div>
                <button type="submit" class="btn btn-full">Detect Faces</button>
            </form>
        </div>
 
        {% if original and processed %}
        <div class="card">
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">{{ face_count }}</div>
                    <div class="stat-label">Faces Detected</div>
                </div>
            </div>
            
            <div class="images">
                <div class="image-box">
                    <h3>Original</h3>
                    <img src="{{ original }}?t={{ timestamp }}">
                </div>
                <div class="image-box">
                    <h3>Detected</h3>
                    <img src="{{ processed }}?t={{ timestamp }}">
                </div>
            </div>
        </div>
        {% endif %}
 
        <p class="host">Host: {{ hostname }}</p>
    </div>
 
    <script>
        function showFileName(input) {
            const fileName = document.getElementById('fileName');
            if (input.files && input.files[0]) {
                fileName.textContent = input.files[0].name;
            }
        }
    </script>
</body>
</html>
"""
 
 
# =============================================================================
# FLASK ROUTES
# =============================================================================
 
@app.route("/")
def home():
    """
    Home route - displays the upload form.
    
    Renders the main HTML template. This is the landing page where
    users upload an image for face detection.
    """
    return render_template_string(HTML, hostname=hostname)
 
 
@app.route("/detect", methods=["POST"])
def detect_route():
    """
    Detection route - handles image upload and runs face detection.
    
    This is the main processing endpoint that:
        1. Receives the uploaded image file from the form
        2. Saves the image to the static folder
        3. Runs YuNet face detection
        4. Returns results with original and processed images
    """
    # Get the uploaded file
    file = request.files.get("file", None)
    
    # Define file paths
    original_path = os.path.join(UPLOAD_FOLDER, "original.jpg")
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    
    # Save uploaded file if one was provided
    if file and file.filename != "":
        file.save(original_path)
    
    # Check that we have an image to process
    if not os.path.exists(original_path):
        return "No image uploaded yet. Please upload an image."
    
    # Run face detection
    face_count = detect_faces(original_path, output_path)
    
    # Render the results page with cache buster
    # The timestamp prevents the browser from caching old images
    return render_template_string(
        HTML,
        original="/static/original.jpg",
        processed="/static/output.jpg",
        face_count=face_count,
        hostname=hostname,
        timestamp=int(time.time())
    )
 
 
@app.route('/static/<path:filename>')
def serve_static(filename):
    """
    Static file server - serves uploaded and processed images.
    
    Used to display the original and processed images in the browser.
    """
    return send_from_directory(UPLOAD_FOLDER, filename)
 
 
# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Get port from environment variable (for cloud deployment)
    # Default to port 80 for local development
    port = int(os.environ.get("PORT", 80))
    
    # Start the Flask server
    # host="0.0.0.0" makes it accessible from outside the container
    app.run(host="0.0.0.0", port=port)