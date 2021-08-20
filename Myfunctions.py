import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Loading the Model
model = load_model('mask_detector.h5')

# Loading the classifier from the file.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# accepted image file extension
UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# initiating the video capturing
camera = cv2.VideoCapture(0)


def allowed_file(filename):
    """ Checks the file format when file is uploaded"""

    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def image_preprocessing(frame):
    path = "static/" + str(frame)
    frame = cv2.imread(path)
    faces_detected = face_cascade.detectMultiScale(frame, 1.2, 7, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces_detected) == 0:
        print("face not detected")

    else:
        faces_images = []
        for (x, y, w, h) in faces_detected:
            cropped_faces = frame[y:y + h, x:x + w]
            cropped_faces = cv2.cvtColor(cropped_faces, cv2.COLOR_BGR2RGB)
            cropped_faces = cv2.resize(cropped_faces, (224, 224))
            cropped_faces = img_to_array(cropped_faces)
            faces_images.append(cropped_faces)

        faces_images = np.array(faces_images)
        faces = preprocess_input(faces_images)
        predictions = model.predict(faces)
        return predictions, frame, faces_detected


def predictions_results(predictions, frame, faces_detected, filename):
    correct_mask_count = []
    incorrect_mask_count = []
    no_mask_count = []

    i = 0
    for pred in predictions:

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
        cv2.rectangle(frame, (x, y + 20), (x + 5 + w, y + h + 15), color, 2)
        cv2.putText(frame, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        i += 1

    cv2.imwrite(f"static\{filename}", frame)
    face_count = len(no_mask_count) + len(incorrect_mask_count) + len(correct_mask_count)
    no_masks = len(no_mask_count)
    corrects_masks = len(correct_mask_count)
    incorrects_masks = len(incorrect_mask_count)

    return face_count, no_masks, corrects_masks, incorrects_masks
