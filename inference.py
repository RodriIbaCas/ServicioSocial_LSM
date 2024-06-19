import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
import time

# Carga el modelo guardado
model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Diccionario para mapear las etiquetas a las letras
labels_dict = {0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 
               5 : 'F', 6 : 'G', 7 : 'H', 8 : 'I', 9 : 'J', 
               10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 
               15 : 'P', 16 : 'Q', 17 : 'R', 18 : 'S', 
               19 : 'T', 20 : 'U', 21 : 'V', 22 : 'W', 23 : 'X', 
               24 : 'Y', 25 : 'Z'}

# Variables para almacenar la letra detectada y el tiempo de detección
detected_letter = None
start_time = None

# Lista para almacenar las letras detectadas
detected_letters = []

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

        if detected_letter == predicted_label:
            if time.time() - start_time >= 3:
                detected_letters.append(predicted_label)
                detected_letter = None
                start_time = None
        else:
            detected_letter = predicted_label
            start_time = time.time()
        
        cv.putText(frame, predicted_label, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv.waitKey(1) & 0xFF == ord('s'):
        with open('detected_letters.txt', 'w') as f:
            f.write(''.join(detected_letters))
        detected_letters = []

cap.release()
cv.destroyAllWindows()