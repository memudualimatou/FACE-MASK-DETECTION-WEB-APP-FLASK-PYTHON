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

# Initiating the video framing
image = cv2.VideoCapture("sgg.mp4")
# image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
faces = face_cascade.detectMultiScale(image, 1.2, 7, minSize=(60, 60),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
    face = image[y:y + h, x:x + w]
    cropped_face = face
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)
    pred = model.predict(face)

    IDs = []

    for preds in pred:
        (WithoutMask, CorrectMask, InCorrectMask) = preds
        if max(preds) == CorrectMask:
            label = " Correct Mask"
            color = (0, 255, 0)
            # correct_mask_count += 1
            IDs.append(1)
        elif max(preds) == InCorrectMask:
            label = " Incorrect Mask"
            color = (250, 00, 0)
            # incorrect_mask_count += 1
            IDs.append(2)
        else:
            label = " No Mask"
            color = (0, 0, 255)
            IDs.append(0)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(WithoutMask, CorrectMask, InCorrectMask) * 100)
        # calculate count values

        # filtered_classids = np.take()
        correct_mask_count = [i for i in IDs if i == 1]
        no_mask_count = [i for i in IDs if i == 0]
        incorrect_mask_count = [i for i in IDs if i == 2]

        # Displaying the labels
        cv2.rectangle(image, (x, y + 20), (x + 5 + w, y + h + 15), color, 1)
        cv2.putText(cropped_face, label, (-7, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        face_count = len(no_mask_count) + len(incorrect_mask_count) + len(correct_mask_count)
        text = "FaceCount: {}   NoMaskCount: {}   CorrectMaskCount: {}  InCorrectMaskCount: {}".format(
            face_count, len(no_mask_count), len(correct_mask_count), len(incorrect_mask_count))
        cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
