import cv2
import gdown
import os
from datetime import datetime

# Google Drive file ID
file_id = '1_ND5on55ypLGe0Rbuw6vbxfoN_xVLz_3'
url = f'https://drive.google.com/uc?id={file_id}'
output_path = 'haarcascade_frontalface_default.xml'

# Download only if not already present
if not os.path.exists(output_path):
    gdown.download(url, output_path, quiet=False)

# Load the classifier
face_cascade = cv2.CascadeClassifier(output_path)


def detect_faces(scale_factor, min_neighbors, box_color):
    cap = cv2.VideoCapture(0)

    # Convert hex color to BGR
    box_color = tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        cv2.imshow('Face Detection', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            # Save the frame with detected faces
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'detected_faces_{timestamp}.png'
            cv2.imwrite(filename, frame)

    cap.release()
    cv2.destroyAllWindows()
