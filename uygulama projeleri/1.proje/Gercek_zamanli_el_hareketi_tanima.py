import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Hareket frekanslarını takip etmek için sayaç
gesture_counter = Counter()

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def detect_gesture(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)
        defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))
        
        if defects is not None:
            count_defects = 0
            for i in range(defects.shape[0]):
                _, _, _, d = defects[i, 0]
                if d > 10000:
                    count_defects += 1
            
            gestures = {0: "Fist", 1: "One Finger", 2: "Two Fingers", 3: "Three Fingers", 4: "Four Fingers", 5: "Open Palm"}
            detected_gesture = gestures.get(count_defects, "Unknown")
            gesture_counter[detected_gesture] += 1
            return detected_gesture
    return "No Gesture"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    mask = process_frame(frame)
    gesture = detect_gesture(mask)
    
    cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Seaborn ile hareket frekanslarını görselleştir
plt.figure(figsize=(8, 5))
sns.barplot(x=list(gesture_counter.keys()), y=list(gesture_counter.values()))
plt.xlabel("Gesture Type")
plt.ylabel("Frequency")
plt.title("Hand Gesture Frequency Analysis")
plt.show()

