import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

while True:
    # Start video stream
    success, image = cap.read()
    if not success:
        print("Video stream couldn't be started")
        break

    # Convert the BGR image to RGB and detect the landmarks
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # Draw landmarks on the image
    if results.multi_hand_landmarks:
        for handLM in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLM, mpHands.HAND_CONNECTIONS)
            # Get the location of the different landmarks in pixels
            height, width, _ = image.shape
            posLandmarks = {}
            for id, landmark in enumerate(handLM.landmark):
                px, py = int(width * landmark.x), int(height * landmark.y)
                posLandmarks[id] = px, py

    # Show video stream with landmarks‚
    cv2.imshow('HandTracker', image)

    # Stop in case you enter esc
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
