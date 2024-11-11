import streamlit as st
import cv2
import torch
import tempfile
import time
from PIL import Image
import numpy as np
from ultralytics import YOLOv10  # Load YOLOv10 model

# Initialize YOLOv10 model
model = YOLOv10("weights/yolov10n.pt")

# COCO class names for object detection
cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                  "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                  "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                  "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
                  "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Streamlit interface setup
st.title("YOLOv10 Object Detection and Tracking")

# Create a sidebar for input options
input_source = st.sidebar.selectbox("Select Input Source", ("Camera", "Image", "Video"))

# Process the selected input source
if input_source == "Camera":
    st.write("Using Camera")
    run = st.checkbox('Start Camera')

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture from camera. Make sure your camera is connected.")
            break

        # Detect objects in the frame
        results = model.predict(frame, conf=0.25)
        detected_objects = []

        # Extract and display detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = cocoClassNames[class_id]
                conf = box.conf[0]

                # Append detected object names
                detected_objects.append(f"{label} ({conf:.2f})")

                # Draw bounding boxes on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the processed frame
        FRAME_WINDOW.image(frame, channels="BGR")

        # Display the names of detected objects
        st.write("Detected Objects: ", detected_objects)

    cap.release()

elif input_source == "Image":
    st.write("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read and convert the uploaded image
        image = np.array(Image.open(uploaded_image))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Detect objects in the image
        results = model.predict(image, conf=0.25)
        detected_objects = []

        # Extract and display detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = cocoClassNames[class_id]
                conf = box.conf[0]

                # Append detected object names
                detected_objects.append(f"{label} ({conf:.2f})")

                # Draw bounding boxes on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the processed image
        st.image(image, caption='Processed Image', use_column_width=True)

        # Display the names of detected objects
        st.write("Detected Objects: ", detected_objects)

elif input_source == "Video":
    st.write("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Create video capture object
        cap = cv2.VideoCapture(video_path)
        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to process video")
                break

            # Detect objects in the video frame
            results = model.predict(frame, conf=0.25)
            detected_objects = []

            # Extract and display detection results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    label = cocoClassNames[class_id]
                    conf = box.conf[0]

                    # Append detected object names
                    detected_objects.append(f"{label} ({conf:.2f})")

                    # Draw bounding boxes on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display the processed video frame
            FRAME_WINDOW.image(frame, channels="BGR")

            # Display the names of detected objects
            st.write("Detected Objects: ", detected_objects)

        cap.release()

