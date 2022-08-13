import cv2
import mediapipe as mp
import osascript

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

while True:
    # start video stream
    success, image = cap.read()
    if not success:
        print("Video stream couldn't be started")
        break

    # convert the BGR image to RGB and detect the landmarks
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_RGB)

    # draw landmarks on the image
    pos_landmarks = {}
    if results.multi_hand_landmarks:
        for hand_LM in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_LM, mp_hands.HAND_CONNECTIONS)
            # get the location of the different landmarks in pixels
            height, width, _ = image.shape

            for id, landmark in enumerate(hand_LM.landmark):
                px, py = int(width * landmark.x), int(height * landmark.y)
                pos_landmarks[id] = px, py

    if pos_landmarks:
        # draw a line between thumb and index finger when other fingers closed
        thumb = pos_landmarks[4]
        index = pos_landmarks[8]
        if pos_landmarks[12][1] > pos_landmarks[10][1] and pos_landmarks[16][1] > pos_landmarks[14][1] \
                and pos_landmarks[20][1] > pos_landmarks[18][1]:
            cv2.line(image, thumb, index, (0, 255, 0), 5)

            # get length of the line between the fingers
            max = 160
            offset = 16
            current_length = cv2.norm(thumb, index) - offset
            volume = (current_length / max) * 100

            # check boundaries
            if volume < 0:
                volume = 0
            elif volume > 100:
                volume = 100

            # change volume on mac
            osascript.osascript("set volume output volume {}".format(volume))

            # get the current volume
            code, out, err = osascript.run("output volume of (get volume settings)")
            cv2.putText(image, f'Volume: {out}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # show video stream with landmarks
    cv2.imshow('HandTracker', image)

    # stop in case you enter esc
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
