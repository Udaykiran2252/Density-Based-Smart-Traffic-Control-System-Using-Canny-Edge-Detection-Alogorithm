import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("C:/Users/B.UDAY KIRAN/OneDrive/Dokumen/Density Based Smart Traffic Control System Using Canny Edge Detection Algorithm/WhatsApp Video 2025-06-19 at 10.24.49_61df025f.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

output_width, output_height = 960, 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("detected_output_updated.mp4", fourcc, 20.0, (output_width, output_height))

# Skip points list
skip_points = []

cv2.namedWindow("Smart Traffic System", cv2.WINDOW_NORMAL)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        skip_points.append((x, y))

cv2.setMouseCallback("Smart Traffic System", on_mouse)

# Signal logic
signal_state = "left"
signal_timer = 30
last_switch_time = time.time()

def get_density_label_by_vehicle_count(count):
    return "High" if count > 10 else "Low"

# Processing video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))
    height, width = frame.shape[:2]
    fgmask = fgbg.apply(frame)
    motion_mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Split edges into halves
    mid_x = width // 2
    left_edges = edges[:, :mid_x]
    right_edges = edges[:, mid_x:]

    left_edge_count = cv2.countNonZero(left_edges)
    right_edge_count = cv2.countNonZero(right_edges)

    left_area = left_edges.shape[0] * left_edges.shape[1]
    right_area = right_edges.shape[0] * right_edges.shape[1]

    left_density = (left_edge_count / left_area) * 100
    right_density = (right_edge_count / right_area) * 100

    # Object detection
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    left_total = right_total = 0

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        label = model.names[int(cls)]
        if label not in ['car', 'truck', 'bus', 'motorcycle']:
            continue

        if any(x1 <= px <= x2 and y1 <= py <= y2 for (px, py) in skip_points):
            continue

        cx = (x1 + x2) // 2
        region = motion_mask[y1:y2, x1:x2]
        motion_pixels = cv2.countNonZero(region)
        is_moving = motion_pixels > ((x2 - x1) * (y2 - y1)) * 0.1

        color = (0, 255, 0) if is_moving else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if cx < mid_x:
            left_total += 1
        else:
            right_total += 1

    current_time = time.time()
    remaining_time = int(signal_timer - (current_time - last_switch_time))

    if remaining_time <= 0:
        if signal_state == "left":
            signal_state = "right"
            signal_timer = 60 if right_total > 6 else 30
        else:
            signal_state = "left"
            signal_timer = 60 if left_total > 6 else 30
        last_switch_time = current_time
        remaining_time = signal_timer

    # Manual switching
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):  # Force left green
        signal_state = "left"
        signal_timer = 30
        last_switch_time = time.time()
    elif key == ord('n'):  # Force right green
        signal_state = "right"
        signal_timer = 30
        last_switch_time = time.time()
    elif key == 27:  # ESC
        break

    # Signal colors
    left_color = (0, 255, 0) if signal_state == "left" else (0, 0, 255)
    right_color = (0, 255, 0) if signal_state == "right" else (0, 0, 255)

    # Draw info
    cv2.putText(frame, "LEFT SIGNAL", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
    cv2.putText(frame, f"[{'GREEN' if signal_state == 'left' else 'RED'}]", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
    cv2.putText(frame, f"Time: {remaining_time}s", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
    cv2.putText(frame, f"Vehicle Count: {left_total}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
    cv2.putText(frame, f"Edge Density: {left_density:.2f}%", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
    cv2.putText(frame, f"Density Level: {get_density_label_by_vehicle_count(left_total)}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)

    cv2.putText(frame, "RIGHT SIGNAL", (width - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
    cv2.putText(frame, f"[{'GREEN' if signal_state == 'right' else 'RED'}]", (width - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    cv2.putText(frame, f"Time: {remaining_time}s", (width - 280, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    cv2.putText(frame, f"Vehicle Count: {right_total}", (width - 300, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
    cv2.putText(frame, f"Edge Density: {right_density:.2f}%", (width - 300, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
    cv2.putText(frame, f"Density Level: {get_density_label_by_vehicle_count(right_total)}", (width - 300, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)

    # Skipped points
    for px, py in skip_points:
        cv2.circle(frame, (px, py), 5, (255, 255, 0), -1)

    out.write(frame)
    cv2.imshow("Smart Traffic System", frame)

cap.release()
out.release()
cv2.destroyAllWindows()