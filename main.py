#copyleft @2024 Arco zp
# feel free to modify this code as you guys need
# happy coding!!
import cv2
import mediapipe as mp
import time

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Set the desired frame width and height
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize MediaPipe Hands and Face Detection modules
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
hands = mp_hands.Hands()
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables for FPS calculation
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results_hands = hands.process(rgb_frame)
    
    if results_hands.multi_hand_landmarks:
        for landmarks in results_hands.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Process the frame with MediaPipe Face Detection
    results_face = face_detection.process(rgb_frame)
    
    if results_face.detections:
        for detection in results_face.detections:
            bbox_c = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), int(bbox_c.width * iw), int(bbox_c.height * ih)
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)
    
    # Calculate FPS and display on the frame
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame with hand landmarks, face detection, and FPS
    cv2.imshow("Hand and Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
