import cv2
import mediapipe as mp
import webbrowser
import time


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) 
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


last_finger_count = None
last_action_time = 0  
cooldown = 2  

current_gesture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.flip(frame, 1)

   
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    results = hands.process(rgb_frame)

   
    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
            for tip in tips:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    finger_count += 1

            
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
                finger_count += 1

    
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    current_time = time.time()
    if finger_count != last_finger_count and (current_time - last_action_time) > cooldown:
        if finger_count == 2 and current_gesture != 2:
            webbrowser.open_new_tab("https://www.youtube.com")
            current_gesture = 2
            last_action_time = current_time
        elif finger_count == 3 and current_gesture != 3:
            webbrowser.open_new_tab("https://www.google.com")  
            current_gesture = 3
            last_action_time = current_time
        elif finger_count == 4 and current_gesture != 4:
            webbrowser.open_new_tab("https://www.github.com")
            current_gesture = 4
            last_action_time = current_time
        elif finger_count == 5 and current_gesture != 5:
            webbrowser.open_new_tab("https://www.chatgpt.com")
            current_gesture = 5
            last_action_time = current_time
        elif finger_count == 1:
            current_gesture = None  

    
    last_finger_count = finger_count

    
    cv2.imshow('Hand Gesture Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()