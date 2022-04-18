import cv2
import mediapipe as mp
import osascript

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
    posLandmarks = {}
    if results.multi_hand_landmarks:
        for handLM in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLM, mpHands.HAND_CONNECTIONS)
            # Get the location of the different landmarks in pixels
            height, width, _ = image.shape

            for id, landmark in enumerate(handLM.landmark):
                px, py = int(width * landmark.x), int(height * landmark.y)
                posLandmarks[id] = px, py

    if posLandmarks:
        # Draw a line between thumb and index finger when other fingers closed
        thumb = posLandmarks[4]
        index = posLandmarks[8]
        if posLandmarks[12][1] > posLandmarks[10][1] and posLandmarks[16][1] > posLandmarks[14][1] \
                and posLandmarks[20][1] > posLandmarks[18][1]:
            cv2.line(image, thumb, index, (0, 255, 0), 5)

            # Get length of the line between the fingers
            max = 160
            offset = 16
            currentLength = cv2.norm(thumb, index) - offset
            volume = (currentLength / max) * 100

            # Check boundaries
            if volume < 0:
                volume = 0
            elif volume > 100:
                volume = 100

            # Change volume on mac
            osascript.osascript("set volume output volume {}".format(volume))

            # Get the current volume
            code, out, err = osascript.run("output volume of (get volume settings)")
            cv2.putText(image, f'Volume: {out}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show video stream with landmarks‚
    cv2.imshow('HandTracker', image)

    # Stop in case you enter esc
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
