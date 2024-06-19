import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carga el archivo pickle
file = pickle.load(open('data.pickle', 'rb'))

# Función para paddear las sublistas y las listas principales
def pad_sequence(seq, max_len, max_inner_len):
    padded_seq = [item + [0] * (max_inner_len - len(item)) for item in seq]
    padded_seq += [[0] * max_inner_len] * (max_len - len(padded_seq))
    return padded_seq

# Encuentra la longitud máxima de las listas y las sublistas en 'data'
max_len = max(len(item) for item in file['data'])
max_inner_len = max(len(subitem) for item in file['data'] for subitem in item)

# Paddea las listas en 'data' para que todas tengan la misma longitud
data = np.array([pad_sequence(item, max_len, max_inner_len) for item in file['data']])

# Aplana las listas internas para tener una estructura uniforme
data = data.reshape((data.shape[0], -1))

# Convertir las etiquetas a array de numpy
labels = np.asarray(file['labels'])

# División de los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) 

# Entrenamiento del modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predicción y evaluación del modelo
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)

print(f'{accuracy * 100 }% de precisión en el conjunto de prueba')


f = open('model.pickle', 'wb')
pickle.dump({'model': model}, f)
f.close()