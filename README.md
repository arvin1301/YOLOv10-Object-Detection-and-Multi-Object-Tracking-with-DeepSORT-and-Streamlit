# YOLOv10-Object-Detection-and-Multi-Object-Tracking-with-DeepSORT-and-Streamlit


This project leverages YOLOv10 for accurate object detection and DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) to enable robust multi-object tracking. It provides an intuitive web interface via Streamlit, allowing users to perform real-time object detection and tracking using their camera, or to analyze uploaded images and videos.



# Key Features
-Real-Time Detection and Tracking: Detects and continuously tracks objects in real time from a camera feed.
-Image and Video Processing: Processes and tracks objects in uploaded images and videos.
-Streamlit-Based UI: Streamlit provides an easy-to-use web interface with a sidebar for selecting input sources.
-DeepSORT Integration: DeepSORT assigns unique IDs to each detected object, maintaining consistent tracking across frames even when objects overlap or reappear.
-Object Identification and Confidence Scores: Displays each object’s class name and detection confidence level.
-FPS Counter: Provides real-time FPS (frames per second) information to monitor processing speed.



# Dependencies
Make sure to install the following libraries:
-opencv-python: For real-time video processing.
-torch: For model inference with YOLOv10 and DeepSORT.
-ultralytics: To load and use the YOLOv10 model.
-streamlit: For building the web interface.
-Pillow: For handling image uploads in Streamlit.
-numpy: For data processing and matrix operations.
-DeepSORT (or custom implementation): For object tracking across frames.



# Install the required libraries with:
-pip install opencv-python torch ultralytics streamlit pillow numpy deepsort


# Project Setup
Clone the Repository:
git clone https://github.com/yourusername/YOLOv10-DeepSORT-Streamlit.git
cd YOLOv10-DeepSORT-Streamlit



# Download YOLOv10 Model Weights:
Place the YOLOv10 model weights (yolov10n.pt) in the weights/ folder.



# Run the Application:
Launch the app with:
-streamlit run app.py


# Usage Guide
1. Selecting the Input Source
The sidebar allows you to choose between three input sources:


-Camera: Real-time detection and tracking through your device’s camera.
-Image: Upload an image file (JPEG, PNG), and the application will display detected and tracked objects.
-Video: Upload a video file (MP4, AVI, MOV), which will be processed frame-by-frame.


2. Object Detection and Tracking Process
Each input type processes the data as follows:



-Detection: YOLOv10 detects objects and draws bounding boxes around them.
-Tracking: DeepSORT assigns a unique ID to each detected object, which persists as long as the object remains in the frame. This is especially useful for tracking objects across consecutive frames in video feeds.
-Labeling and Confidence: Each detected object’s class label and confidence score are displayed on the bounding box.
-FPS: The FPS counter shows the processing speed, which may vary based on system performance and input type.


3. Output Display
-Bounding Boxes: Detected objects are enclosed in bounding boxes with labels and confidence scores.
-Object IDs: Unique IDs assigned by DeepSORT make it easy to track individual objects, even when they move within the frame.
-Detected Objects List: A list of detected object names with confidence levels is displayed in the Streamlit UI.


4. Real-Time Controls
-Start/Stop Camera: For the "Camera" option, a checkbox in Streamlit allows you to start or stop the camera feed.
-Exit Video: For video input, you can terminate the playback by pressing the “1” key in the OpenCV window.


# Integration of YOLOv10 and DeepSORT
This project combines YOLOv10’s advanced detection capabilities with DeepSORT’s robust tracking. YOLOv10 detects objects in each frame, while DeepSORT maintains object identities across frames, enabling seamless tracking for applications like surveillance, traffic monitoring, and other multi-object tracking scenarios.

# Project Structure

├── app.py                     # Streamlit application entry point
├── utils/
│   └── object_tracking.py     # DeepSORT implementation and helper functions
├── weights/                   # Folder for storing YOLOv10 model weights
├── Resources/                 # Directory containing sample videos for testing
└── README.md                  # Project README file


# Example Outputs
-Camera Feed: Real-time tracking and detection with labeled bounding boxes and FPS counter.
-Image Upload: Display of bounding boxes, object IDs, and detected object list.
-Video Upload: Frame-by-frame detection and tracking with persistent IDs.


# Troubleshooting
-Low FPS: Reduce input resolution or adjust confidence threshold to improve FPS.
-Camera Not Detected: Ensure your camera is properly connected. Restart the app if necessary.



# Screenshot:
![06 11 2024_16 34 29_REC](https://github.com/user-attachments/assets/a7110232-2563-4725-9b18-b76ef0a1764e)
![06 11 2024_16 34 04_REC](https://github.com/user-attachments/assets/0a3f47d9-65c5-4e51-8a20-a3b7b1c701ba)
![06 11 2024_16 26 49_REC](https://github.com/user-attachments/assets/5c16247d-704f-4855-847e-73d569f5ff00)

