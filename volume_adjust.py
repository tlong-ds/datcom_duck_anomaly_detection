# !/usr/bin/env python

"""
Day 1: Introduction to Image Processing + OpenCV
"""

import cv2
import mediapipe as mp
import numpy as np
import subprocess

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)

prev_vol = -1   # track last volume set
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (640, 480))  # reduce resolution
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process every 3rd frame only (~10 FPS for hand detection)
    if frame_count % 3 == 0:
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw only landmarks you need
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)   # thumb tip
                x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)   # index tip

                # Draw key points
                cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Distance between fingers  
                length = np.hypot(x2 - x1, y2 - y1)

                # Map distance [20,200] → volume [0,100]
                vol_percent = int(np.interp(length, [20, 200], [0, 100]))

                # Only update if changed significantly
                if abs(vol_percent - prev_vol) > 2:
                    subprocess.run(
                        ["osascript", "-e", f"set volume output volume {vol_percent}"],
                        capture_output=True
                    )
                    prev_vol = vol_percent

                # Draw volume bar
                vol_bar = np.interp(length, [20, 200], [400, 150])
                cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f'{vol_percent} %', (40, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Show smooth video (30 FPS)
    cv2.imshow("Optimized Hand Volume Control (macOS)", frame)