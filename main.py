import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------- AUDIO SETUP -----------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol, _ = volume.GetVolumeRange()

# ---------------------- MEDIAPIPE SETUP ----------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# ---------------------- CAMERA ----------------------
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

draw_points = []
mode = "mouse"   # mouse / draw / volume

def detect_fingers(lm_list):
    """Returns which fingers are open."""
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if lm_list[4][0] > lm_list[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips[1:]:
        if lm_list[tip][1] < lm_list[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers = detect_fingers(lm_list)
            x1, y1 = lm_list[8]   # Index tip
            x2, y2 = lm_list[4]   # Thumb tip

            # ---------------- MODE SWITCH -----------------
            if fingers == [0, 1, 0, 0, 0]:     # Only index open
                mode = "mouse"

            elif fingers == [0, 1, 1, 0, 0]:   # Index + middle
                mode = "draw"

            elif fingers == [1, 1, 0, 0, 0]:   # Thumb + index
                mode = "volume"

            # ------------------- MOUSE CONTROL -------------------
            if mode == "mouse":
                mouse_x = np.interp(x1, [0, w], [0, screen_w])
                mouse_y = np.interp(y1, [0, h], [0, screen_h])
                pyautogui.moveTo(mouse_x, mouse_y, duration=0)

                # Left Click when index + thumb pinch
                distance = math.hypot(x2 - x1, y2 - y1)
                if distance < 40:
                    pyautogui.click()

                cv2.putText(img, "MODE: MOUSE", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # ------------------- DRAW MODE -------------------
            if mode == "draw":
                draw_points.append((x1, y1))
                for i in range(1, len(draw_points)):
                    cv2.line(img, draw_points[i - 1], draw_points[i], (0, 0, 255), 4)

                cv2.putText(img, "MODE: DRAW", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # ------------------- VOLUME CONTROL -------------------
            if mode == "volume":
                distance = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(distance, [20, 200], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)

                bar = np.interp(distance, [20, 200], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(bar)), (85, 400), (0, 255, 0), cv2.FILLED)

                cv2.putText(img, "MODE: VOLUME", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("AI Hand Controller", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

