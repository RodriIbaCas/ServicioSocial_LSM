import os
import cv2 

DATA_DIR = './data'
VIDEOS_DIR = './videos'  # Directorio donde se encuentran los videos
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26  # Número de clases
dataset_size = 150  # Número de imágenes por clase por video
videos_per_class = 2  # Número de videos por clase

# Función para capturar frames de un video y guardarlos en un directorio
def capture_frames_from_video(video_path, class_id, dataset_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    counter = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // dataset_size)  # Intervalo para capturar frames proporcionalmente

    while counter < dataset_size:
        cap.set(cv2.CAP_PROP_POS_FRAMES, counter * interval)
        ret, frame = cap.read()
        if not ret:
            print(f"End of video {video_path}")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(class_id), '{}_{}.jpg'.format(os.path.basename(video_path), counter)), frame)
        counter += 1

    cap.release()

# Procesa los videos en el directorio cambiando de clase al terminar cada par de videos
video_files = [f for f in os.listdir(VIDEOS_DIR) if os.path.isfile(os.path.join(VIDEOS_DIR, f))]

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    for k in range(videos_per_class):
        video_index = j * videos_per_class + k
        if video_index < len(video_files):
            video_path = os.path.join(VIDEOS_DIR, video_files[video_index])
            print(f"Processing video {video_path} for class {j}")
            capture_frames_from_video(video_path, j, dataset_size)
        else:
            print(f"No more videos to process for class {j}")
            break

cv2.destroyAllWindows()
