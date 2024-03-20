import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import time

# Function to generate video frames
def get_video_frames_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

class Detections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id

def box_annotator(frame, detections, labels):
    for i in range(len(detections.xyxy)):
        xyxy = detections.xyxy[i]
        class_id = detections.class_id[i]
        confidence = detections.confidence[i]
        label = labels[i]

        # Extract coordinates
        x1, y1, x2, y2 = xyxy.astype(int)
        # Draw bounding box
        color = (0, 255, 0)  # Green color for bounding box
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Add label and confidence
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)

    return frame

# Define source video path and model
SOURCE_VIDEO_PATH = "D:\\HACKATHONS\\Datathon\\try4\\test_video2.mp4"
MODEL = "yolov8x.pt"

# Initialize YOLO model
model = YOLO(MODEL)
model.fuse()

# Get class names dictionary
CLASS_NAMES_DICT = model.model.names

# Class IDs of interest
VEHICLE_CLASS_ID = [2, 3, 5, 7]
PERSON_CLASS_ID = [0]

# Initialize trackers for vehicles and persons
vehicle_trackers = defaultdict(list)
person_trackers = defaultdict(list)

# Initialize previous detections for vehicles and persons
prev_vehicle_detections = defaultdict(list)
prev_person_detections = defaultdict(list)

# Create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

# Congestion detection variables
congestion_threshold = 3
congestion_frame_threshold = 2
congestion_counter = defaultdict(int)
congestion_frames_counter = defaultdict(int)
congestion_saved_counter = defaultdict(int)

# Process video frames
frame_count = 0
for frame in generator:
    frame_count += 1
    if frame_count % 20 == 0:
        # Model prediction on single frame and conversion to supervision Detections
        results = model(frame)
        detections = results[0].boxes
        detections = Detections(
            xyxy=detections.xyxy.cpu().numpy(),
            confidence=detections.conf.cpu().numpy(),
            class_id=detections.cls.cpu().numpy().astype(int)
        )

        # Format custom labels
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)
        ]

        # Annotate frame
        frame = box_annotator(frame=frame, detections=detections, labels=labels)

        # Congestion detection
        for class_id, xyxy in zip(detections.class_id, detections.xyxy):
            # Check if class is vehicle or person
            if class_id in VEHICLE_CLASS_ID or class_id in PERSON_CLASS_ID:
                # Convert xyxy to centroid
                centroid_x = (xyxy[0] + xyxy[2]) / 2
                centroid_y = (xyxy[1] + xyxy[3]) / 2

                # Update congestion counter
                congestion_counter[(class_id, centroid_x, centroid_y)] += 1

                # Check if congestion threshold is reached
                if congestion_counter[(class_id, centroid_x, centroid_y)] >= congestion_threshold:
                    congestion_frames_counter[(class_id, centroid_x, centroid_y)] += 1

                    # Check if congestion frame threshold is reached
                    if congestion_frames_counter[(class_id, centroid_x, centroid_y)] >= congestion_frame_threshold:
                        # Check if congestion has already been saved for this entity three times
                        if congestion_saved_counter[(class_id, centroid_x, centroid_y)] < 3:
                            # Mark congestion with circle
                            farthest_distance = 0
                            for other_class_id, other_xyxy in zip(detections.class_id, detections.xyxy):
                                if class_id != other_class_id:
                                    other_centroid_x = (other_xyxy[0] + other_xyxy[2]) / 2
                                    other_centroid_y = (other_xyxy[1] + other_xyxy[3]) / 2
                                    distance = np.sqrt((centroid_x - other_centroid_x)**2 + (centroid_y - other_centroid_y)**2)
                                    farthest_distance = max(farthest_distance, distance)

                            # Mark congestion with yellow circle
                            color = (0, 255, 255)  # Yellow color for congestion circle
                            thickness = 2
                            cv2.circle(frame, (int(centroid_x), int(centroid_y)), int(farthest_distance / 5), color, thickness)

                            # Draw red lines connecting centers of all rectangular frames inside the congestion circle
                            for other_class_id, other_xyxy in zip(detections.class_id, detections.xyxy):
                                if class_id != other_class_id:
                                    other_centroid_x = (other_xyxy[0] + other_xyxy[2]) / 2
                                    other_centroid_y = (other_xyxy[1] + other_xyxy[3]) / 2
                                    distance = np.sqrt((centroid_x - other_centroid_x)**2 + (centroid_y - other_centroid_y)**2)
                                    if distance <= (farthest_distance / 5):
                                        cv2.line(frame, (int(centroid_x), int(centroid_y)), (int(other_centroid_x), int(other_centroid_y)), (0, 0, 255), thickness)

                            # Save frame with timestamp
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            # save_path = f"identified_congestion/{timestamp}.jpg"
                            save_path = f"identified_congestion/pic.jpg"
                            cv2.imwrite(save_path, frame)

                            # Increment congestion saved counter
                            congestion_saved_counter[(class_id, centroid_x, centroid_y)] += 1

                            print(f"Congestion detected at {timestamp}")

                        # Reset congestion frames counter
                        congestion_frames_counter[(class_id, centroid_x, centroid_y)] = 0

        # Display the frame in a window
        cv2.imshow('Frame', frame)

        # Set a small delay between frames for smoother playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture
cv2.destroyAllWindows()
