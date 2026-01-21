import sys
print(sys.executable)
import cv2
import mediapipe as mp

# --Load images
happyw = cv2.imread("./happywink_emo.jpg")
ok=cv2.imread("./ok_emo.jpg")
side = cv2.imread("./side_emo.jpg")
smilet=cv2.imread("./smileteeth_emo.jpg")
ltalk=cv2.imread("./talkleft_emo.jpg")
rtalk=cv2.imread("./talkright_emo.jpg")
tweed=cv2.imread("./tumbleweed.gif")
///
import cv2
import numpy as np

print("Starting emotion recognition system...")
print("Make sure your webcam is connected!")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load emotion overlay images (optional)
try:
    happyw = cv2.imread("./happywink_emo.jpg")
    ok = cv2.imread("./ok_emo.jpg")
    side = cv2.imread("./side_emo.jpg")
    smilet = cv2.imread("./smileteeth_emo.jpg")
    ltalk = cv2.imread("./talkleft_emo.jpg")
    rtalk = cv2.imread("./talkright_emo.jpg")
    print("✓ Emotion images loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Emotion images not loaded (this is OK): {e}")

def detect_emotion(face_gray, face_color, x, y, w, h):
    """Detect emotion based on facial features"""
    
    # Region of interest for face
    roi_gray = face_gray[y:y+h, x:x+w]
    roi_color = face_color[y:y+h, x:x+w]
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    
    # Detect smile in lower half of face
    smile_roi = roi_gray[int(h/2):h, 0:w]
    smiles = smile_cascade.detectMultiScale(smile_roi, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
    
    # Draw eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Determine emotion
    emotion = "Neutral"
    color = (200, 200, 200)
    
    if len(eyes) >= 2:
        if len(smiles) > 0:
            emotion = "Happy/Smiling"
            color = (0, 255, 0)
            # Draw smile
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, int(h/2)+sy), (sx+sw, int(h/2)+sy+sh), (255, 255, 0), 2)
        else:
            emotion = "Neutral"
            color = (255, 255, 0)
    elif len(eyes) == 1:
        emotion = "Winking"
        color = (255, 0, 255)
    elif len(eyes) == 0:
        emotion = "Eyes Closed"
        color = (0, 165, 255)
    
    return emotion, color

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot access webcam!")
    print("Make sure:")
    print("  1. Your webcam is connected")
    print("  2. No other application is using the webcam")
    print("  3. You've granted camera permissions")
    exit()

print("\n" + "="*50)
print("🎥 EMOTION RECOGNITION STARTED")
print("="*50)
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 's' to take a screenshot")
print("  - Press 'r' to reset")
print("="*50 + "\n")

frame_count = 0

while True:
    # Capture frame
    ret, frame = cap.read()
    
    if not ret:
        print("⚠ Warning: Can't receive frame. Retrying...")
        continue
    
    frame_count += 1
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(100, 100)
    )
    
    # Process each detected face
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Detect emotion
            emotion, color = detect_emotion(gray, frame, x, y, w, h)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw emotion label with background
            label = f"{emotion}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            # Background rectangle for text
            cv2.rectangle(frame, 
                         (x, y - 40), 
                         (x + label_size[0] + 10, y), 
                         color, 
                         -1)
            
            # Emotion text
            cv2.putText(frame, label, 
                       (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 0), 2)
            
            # Draw face count
            cv2.putText(frame, f"Face {len(faces)}", 
                       (x, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
    else:
        # No face detected
        cv2.putText(frame, "No face detected - Please face the camera", 
                   (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255), 2)
    
    # Display info
    height, width = frame.shape[:2]
    
    # Info panel background
    cv2.rectangle(frame, (0, height - 80), (width, height), (0, 0, 0), -1)
    
    # Display controls
    cv2.putText(frame, "Controls: [Q]uit | [S]creenshot | [R]eset", 
               (10, height - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 1)
    
    # Display frame count
    cv2.putText(frame, f"Frame: {frame_count} | Faces: {len(faces)}", 
               (10, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 1)
    
    # Show the frame
    cv2.imshow('Emotion Recognition System', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Quitting...")
        break
    elif key == ord('s'):
        filename = f"screenshot_{frame_count}_{np.random.randint(1000, 9999)}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    elif key == ord('r'):
        frame_count = 0
        print("🔄 Reset frame counter")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("✓ Program ended successfully")
print(f"Total frames processed: {frame_count}")
print("="*50)