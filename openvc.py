import cv2
import os
import time
import math

# Create a folder for saved frames
save_folder = "captured_frames"
os.makedirs(save_folder, exist_ok=True)

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables for tracking
prev_center = None
prev_time = time.time()
direction_text = "Initializing..."
speed_text = "Speed: 0 px/s"

# --- Adjustable Parameters ---
movement_threshold = 15     # Minimum pixel movement to trigger direction
smoothing_factor = 0.7      # 0-1 range; higher = smoother motion tracking
font_scale = 0.8
font_color = (0, 255, 255)
speed_color = (255, 255, 255)
line_thickness = 2

# Start the webcam
cap = cv2.VideoCapture(0)

# Set the desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize smoothed positions
smoothed_cx, smoothed_cy = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        # Select the largest detected face (most likely the primary subject)
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute current center of face
        cx, cy = x + w // 2, y + h // 2

        # Smooth movement (Exponential smoothing)
        if smoothed_cx is None:
            smoothed_cx, smoothed_cy = cx, cy
        else:
            smoothed_cx = int(smoothing_factor * smoothed_cx + (1 - smoothing_factor) * cx)
            smoothed_cy = int(smoothing_factor * smoothed_cy + (1 - smoothing_factor) * cy)

        # Draw center point
        cv2.circle(frame, (smoothed_cx, smoothed_cy), 6, (255, 0, 0), -1)

        # Calculate direction and speed
        if prev_center is not None:
            dx = smoothed_cx - prev_center[0]
            dy = smoothed_cy - prev_center[1]
            curr_time = time.time()
            dt = curr_time - prev_time if curr_time - prev_time != 0 else 1e-5

            distance = math.sqrt(dx ** 2 + dy ** 2)
            speed = distance / dt  # pixels per second

            # Determine direction
            if abs(dx) < movement_threshold and abs(dy) < movement_threshold:
                direction_text = "Stable"
            elif abs(dx) > abs(dy):  # Horizontal movement
                if dx > 0:
                    direction_text = "Moving Right"
                else:
                    direction_text = "Moving Left"
            else:  # Vertical movement
                if dy > 0:
                    direction_text = "Moving Down"
                else:
                    direction_text = "Moving Up"

            speed_text = f"Speed: {speed:.2f} px/s"
            prev_center = (smoothed_cx, smoothed_cy)
            prev_time = curr_time
        else:
            prev_center = (smoothed_cx, smoothed_cy)
            prev_time = time.time()
    else:
        direction_text = "No Face Detected"
        speed_text = ""

    # Display information on screen
    cv2.putText(frame, direction_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, line_thickness)
    cv2.putText(frame, speed_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, speed_color, line_thickness)

    cv2.imshow("Face Direction Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"captured_frame_{timestamp}.png")
        cv2.imwrite(save_path, frame)
        print(f"Frame saved at {save_path}")

cap.release()
cv2.destroyAllWindows()
