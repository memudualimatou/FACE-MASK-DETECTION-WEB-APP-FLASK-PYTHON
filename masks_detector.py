# Importing Libraries
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Loading the Model
model = load_model('mask_detector.h5')

# Loading the face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# function
def face_preprocesssing(faces_detected):
    faces_images = []
    for (x, y, w, h) in faces_detected:
        cropped_faces = frame[y:y + h, x:x + w]
        cropped_faces = cv2.cvtColor(cropped_faces, cv2.COLOR_BGR2RGB)
        cropped_faces = cv2.resize(cropped_faces, (224, 224))
        cropped_faces = img_to_array(cropped_faces)
        faces_images.append(cropped_faces)
    return faces_images


# Initiating the video framing
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    faces_detected = face_cascade.detectMultiScale(frame, 1.2, 7, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    try:
        faces = face_preprocesssing(faces_detected)
        faces = np.array(faces)
        face = preprocess_input(faces)
        preds = model.predict(face)

        correct_mask_count = []
        incorrect_mask_count = []
        no_mask_count = []

        i = 0
        for pred in preds:

            (WithoutMask, CorrectMask, InCorrectMask) = pred
            if max(pred) == CorrectMask:
                label = " Correct Mask"
                color = (0, 255, 0)
                correct_mask_count.append(1)
            elif max(pred) == InCorrectMask:
                label = " Incorrect Mask"
                color = (250, 00, 0)
                incorrect_mask_count.append(2)
            else:
                label = " No Mask"
                color = (0, 0, 255)
                no_mask_count.append(0)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(WithoutMask, CorrectMask, InCorrectMask) * 100)
            (x, y, w, h) = faces_detected[i]
            # Displaying the labels
            cv2.rectangle(frame, (x, y + 20), (x + 5 + w, y + h + 15), color, 1)
            cv2.putText(frame, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            i += 1

        face_count = len(no_mask_count) + len(incorrect_mask_count) + len(correct_mask_count)

        text = "FaceCount: {}   NoMaskCount: {}   CorrectMaskCount: {}  InCorrectMaskCount: {}".format(
            face_count, len(no_mask_count), len(correct_mask_count), len(incorrect_mask_count))
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    except:
        pass

    cv2.imshow('FACE MASK DETECTOR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
