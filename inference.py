import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np

# Carga el modelo guardado
model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}

while True:
    data_aux = []



    ret, frame = cap.read()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

        # Asegúrate de que data_aux tenga la longitud correcta
        if len(data_aux) != 84:
            data_aux += [0] * (84 - len(data_aux))

        # Predice con el modelo
        prediction = model.predict(np.array(data_aux).reshape(1, -1))

        predicted_label = labels_dict[prediction[0]]


        cv.putText(frame, predicted_label, (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)


    cv.imshow('frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
