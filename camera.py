import cv2
import serial
import time

# ======== SERIAL CONNECTION ========
arduino = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2)

# ======== FACE DETECTOR ========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ======== CAMERA SETUP ========
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# ======== PARAMETERS ========
command_delay = 0.1          # seconds between commands
center_tolerance_px = 2      # pixels tolerance before moving
screen_width_deg = 180       # full horizontal screen corresponds to 180°
last_command_time = 0
current_pan_angle = 90       # start at middle

print("Face Tracking System Started")
print("Arduino connected:", arduino.is_open)

def pixel_to_degree(pixel_x, frame_width):
    """Convert x-coordinate to stepper degree (0-180°)"""
    degree = (pixel_x / frame_width) * screen_width_deg
    return degree

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

    frame_center_x = frame.shape[1] // 2
    frame_width = frame.shape[1]

    # Draw center line
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame.shape[0]), (255, 255, 255), 1)

    if len(faces) > 0:
        # Pick largest face
        largest_face = max(faces, key=lambda f: f[2]*f[3])
        x, y, w, h = largest_face
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Draw face and center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 0), -1)
        cv2.line(frame, (frame_center_x, frame.shape[0]//2), (face_center_x, frame.shape[0]//2), (0, 0, 255), 1)

        # Convert pixel to stepper degree
        target_angle = pixel_to_degree(face_center_x, frame_width)

        # Only move if difference is significant
        if abs(target_angle - current_pan_angle) > center_tolerance_px:
            current_time = time.time()
            if (current_time - last_command_time) > command_delay:
                rotation_needed = target_angle - current_pan_angle
                command = f"rotate {-rotation_needed:.1f} degrees\n"
                arduino.write(command.encode())
                print(f"[MOVE] Rotating {rotation_needed:.1f}° to reach {target_angle:.1f}°")
                current_pan_angle = target_angle
                last_command_time = current_time
    else:
        # No face detected, do not move
        cv2.putText(frame, "No face detected - Stepper stopped", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display current pan
    cv2.putText(frame, f"Current Pan: {current_pan_angle:.1f}°", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Tracker 0-180°", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
arduino.close()
cv2.destroyAllWindows()
