# Importing Libraries
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Loading the Model
model = load_model('mask_detector.h5')

# Loading the face
# detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initiating the image
image = cv2.imread("testing faces\Cool1.jpg")
image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_AREA)
faces_detected = face_cascade.detectMultiScale(image, 1.2, 7, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

faces_images = []
for (x, y, w, h) in faces_detected:
    cropped_faces = image[y:y + h, x:x + w]
    cropped_faces = cv2.cvtColor(cropped_faces, cv2.COLOR_BGR2RGB)
    cropped_faces = cv2.resize(cropped_faces, (224, 224))
    cropped_faces = img_to_array(cropped_faces)
    faces_images.append(cropped_faces)

faces_images = np.array(faces_images)
faces = preprocess_input(faces_images)
preds = model.predict(faces)

for pred in preds:
    (WithoutMask, CorrectMask, InCorrectMask) = pred
    if max(pred) == CorrectMask:
        label = " Correct Mask"
        color = (0, 255, 0)
    elif max(pred) == InCorrectMask:
        label = " Incorrect Mask"
        color = (205, 00, 0)

    else:
        label = " No Mask"
        color = (0, 0, 255)

    # include the probability in the label
    label = "{}: {:.2f}%".format(label, max(WithoutMask, CorrectMask, InCorrectMask) * 100)
    cv2.rectangle(image, (x, y + 20), (x + 5 + w, y + h + 15), color, 1)
    cv2.putText(image, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
