import cv2
import numpy as np

print("Starting emotion recognition system...")
print("Loading emotion images...")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load emotion images with detailed diagnostics
emotion_images = {}
image_files = {
    'happy': './happywink_emo.jpg',
    'ok': './ok_emo.jpg',
    'side': './side_emo.jpg',
    'smile': './smileteeth_emo.jpg',
    'talk_left': './talkleft_emo.jpg',
    'talk_right': './talkright_emo.jpg',
}

print("\n" + "="*50)
print("Loading emotion images...")
print("="*50)

for key, filepath in image_files.items():
    img = cv2.imread(filepath)
    emotion_images[key] = img
    if img is not None:
        h, w = img.shape[:2]
        print(f"✓ {key}: {filepath} ({w}x{h})")
    else:
        print(f"✗ {key}: {filepath} - FILE NOT FOUND!")

# Try loading tumbleweed (gif 1st frame)
tumbleweed = cv2.imread("./tumbleweed.gif")

emotion_images['tumbleweed'] = tumbleweed
if tumbleweed is not None:
    h, w = tumbleweed.shape[:2]
    print(f"tumbleweed: loaded ({w}x{h})")
else:
    print(f"tumbleweed: NOT FOUND (tried .gif)")

loaded_count = sum(1 for img in emotion_images.values() if img is not None)
print(f"\nTotal: {loaded_count}/{len(emotion_images)} images loaded")
print("="*50 + "\n")

if loaded_count == 0:
    print("⚠ WARNING: No emotion images found!")
    print("Please make sure image files are in the same folder as this script.")
    print("Current directory:", __file__ if '__file__' in dir() else "unknown")

def resize_image(img, target_height):
    """Resize image maintaining aspect ratio"""
    if img is None:
        return None
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    resized = cv2.resize(img, (new_width, target_height))
    return resized

def create_emotion_panel(current_emotion, panel_width, panel_height):
    """Create a side panel showing the current emotion image"""
    # Create dark gray background panel (easier to see)
    panel = np.full((panel_height, panel_width, 3), 30, dtype=np.uint8)
    
    # Map detected emotion to image key
    emotion_map = {
        'Happy/Smiling': 'smile',
        'Neutral': 'ok',
        'Winking': 'happy',
        'Eyes Closed': 'side',
        'Talking Left': 'talk_left',
        'Talking Right': 'talk_right',
        'No Face': 'tumbleweed'
    }
    
    image_key = emotion_map.get(current_emotion, 'ok')
    emotion_img = emotion_images.get(image_key)
    
    # Add header section with colored background
    cv2.rectangle(panel, (0, 0), (panel_width, 100), (50, 50, 50), -1)
    
    # Add text label at the top
    cv2.putText(panel, "EMOTION:", 
               (10, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    
    cv2.putText(panel, current_emotion, 
               (10, 75), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               0.9, (0, 255, 255), 2)
    
    if emotion_img is not None:
        # Resize emotion image to fit panel (leaving margin for header)
        available_height = panel_height - 120  # Leave space for header
        resized_img = resize_image(emotion_img, available_height)
        
        if resized_img is not None:
            img_h, img_w = resized_img.shape[:2]
            
            # Make sure image fits in panel
            if img_w > panel_width - 20:
                # If too wide, resize based on width instead
                scale = (panel_width - 20) / img_w
                new_width = panel_width - 20
                new_height = int(img_h * scale)
                resized_img = cv2.resize(resized_img, (new_width, new_height))
                img_h, img_w = resized_img.shape[:2]
            
            # Center the image in the panel (below header)
            y_offset = 110 + (panel_height - 110 - img_h) // 2
            x_offset = (panel_width - img_w) // 2
            
            # Place image on panel with boundary check
            if (y_offset >= 0 and x_offset >= 0 and 
                y_offset + img_h <= panel_height and 
                x_offset + img_w <= panel_width):
                panel[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = resized_img
            else:
                # Debug message if image doesn't fit
                cv2.putText(panel, "Image too large", 
                           (10, panel_height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 255), 2)
    else:
        # Show message if image not loaded
        cv2.putText(panel, "Image not found", 
                   (10, panel_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
        cv2.putText(panel, f"Key: {image_key}", 
                   (10, panel_height//2 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
    
    return panel

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
    eye_positions=[]
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        eye_center_x = ex+ew//2
        eye_positions.append(eye_center_x)
    #Calculate if head is turned based on eye position relative to the fac center
    face_center_x=w//2
    head_turned=None

    if len(eye_positions) >= 2:
        #Average eye position
        avg_eye_x=sum(eye_positions) / len(eye_positions)
        offset=avg_eye_x-face_center_x

        if offset > w *0.10: #0.10 small head turns, 0.15 medium sensitivity, 0.20 only obvious turns
            head_turned="right"
        elif offset < -w*0.10:
            head_turned="left"

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
        elif head_turned == "right":
            emotion="Talking Right"
            color=(255,100,255)
        elif head_turned == "left":
            emotion="Talking Left"
            color=(255,165,0)
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
    exit()

print("\n" + "="*50)
print("🎥 EMOTION RECOGNITION WITH IMAGE DISPLAY")
print("="*50)
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 's' to take a screenshot")
print("="*50 + "\n")

frame_count = 0
current_emotion = "No Face"

# Panel dimensions
PANEL_WIDTH = 300

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("⚠ Warning: Can't receive frame. Retrying...")
        continue
    
    frame_count += 1
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    
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
            current_emotion = emotion
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw emotion label
            label = f"{emotion}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            cv2.rectangle(frame, 
                         (x, y - 35), 
                         (x + label_size[0] + 10, y), 
                         color, 
                         -1)
            
            cv2.putText(frame, label, 
                       (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 0), 2)
    else:
        current_emotion = "No Face"
        cv2.putText(frame, "No face detected", 
                   (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255), 2)
    
    # Create emotion panel
    emotion_panel = create_emotion_panel(current_emotion, PANEL_WIDTH, frame_height)
    
    # Combine frame and panel horizontally
    combined_frame = np.hstack((frame, emotion_panel))
    
    # Display info at bottom
    height, width = combined_frame.shape[:2]
    cv2.rectangle(combined_frame, (0, height - 60), (width, height), (0, 0, 0), -1)
    
    cv2.putText(combined_frame, "Controls: [Q]uit | [S]creenshot", 
               (10, height - 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 1)
    
    cv2.putText(combined_frame, f"Frame: {frame_count} | Faces: {len(faces)}", 
               (10, height - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 1)
    
    # Show the combined frame
    cv2.imshow('Emotion Recognition with Image Display', combined_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Quitting...")
        break
    elif key == ord('s'):
        filename = f"emotion_screenshot_{frame_count}.jpg"
        cv2.imwrite(filename, combined_frame)
        print(f"📸 Screenshot saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("✓ Program ended successfully")
print(f"Total frames processed: {frame_count}")
print("="*50)