# MULTI-FACE MASK DETECTION MODEL: The goal of this model is to perfectly detect a person correctly or incorreclty
# wearing a face mask and the one who is not.

# Importing libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
from zipfile import ZipFile
import os
import cv2
from sklearn.model_selection import train_test_split
from multiprocessing import Process

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input


# creating a function to print the project path
def get_all_files(directory):  # function to get all files from directory
    paths = []
    for root, dirs, files in os.walk(directory):
        for f_name in files:
            path = os.path.join(root, f_name)  # get a file and add the total path
            paths.append(path)
    return paths  # Return the file paths


directory = 'C:\MINE\DATA SCIENCE\my datasets\MASKS'
paths = get_all_files(directory)
print(paths)

# creating folder
PATH = {
    "": "",
    "CSV": os.path.join("DETECT MASK 2", "CSV", ),
    "IMAGES": os.path.join("DETECT MASK 2", "IMAGES", ),
    "garbage": os.path.join("DETECT MASK 2", "IMAGES", "images"),
    "train": os.path.join("DETECT MASK 2", "IMAGES", "train"),
    "test": os.path.join("DETECT MASK 2", "IMAGES", "test"),
    "garbage_df": os.path.join("DETECT MASK 2", 'train_images_labels.csv'),
}


# # DATA PREPROCESSING
def unzip():
    print('unzip file')
    with ZipFile('C:\\MINE\\DATA SCIENCE\\my datasets\\MASKS\\images.zip', 'r') as zf:
        zf.extractall(path=PATH['IMAGES'], )
    print('zip completed')


def dataset():
    unzip()

    total_imgs = len(os.listdir(PATH['garbage']))
    print(f"[INFO] {total_imgs} images in garbage folder 'images' ")

    garbage_df = pd.read_csv(PATH["garbage_df"])
    print(f"[INFO] {garbage_df.shape[0]} images in df ")

    print(f"[INFO] Creating folders to arrange train images by label ")
    correct_mask_folder = os.path.join(PATH['train'], "1-CORRECT_MASK")
    no_mask_folder = os.path.join(PATH['train'], "0-NO_MASK")
    incorrect_mask_folder = os.path.join(PATH['train'], "2-INCORRECT_MASK")

    os.makedirs(name=correct_mask_folder, exist_ok=True)
    os.makedirs(name=no_mask_folder, exist_ok=True)
    os.makedirs(name=incorrect_mask_folder, exist_ok=True)

    no_mask_filenames = garbage_df[garbage_df.target == 0].image.tolist()
    correct_mask_filenames = garbage_df[garbage_df.target == 1].image.tolist()
    incorrect_mask_filenames = garbage_df[garbage_df.target == 2].image.tolist()

    # Parallel moving
    print(f"[INFO] Moving NO MASK images ")
    _ = []
    for name in no_mask_filenames:
        try:
            p = Process(target=shutil.move, args=(os.path.join(PATH['garbage'], name), no_mask_folder))
            p.start()
            _ += [p]
        except:
            pass

    print(f"[INFO] Moving INCORRECT MASK images ")
    for name in incorrect_mask_filenames:
        try:
            p = Process(target=shutil.move, args=(os.path.join(PATH['garbage'], name), incorrect_mask_folder))
            p.start()
            _ += [p]
        except:
            pass

    print(f"[INFO] Moving CORRECT MASK images ")
    for name in correct_mask_filenames:
        try:
            p = Process(target=shutil.move, args=(os.path.join(PATH['garbage'], name), correct_mask_folder))
            p.start()
            _ += [p]
        except:
            pass
    [p.join() for p in _]

    print(f"[INFO] Rename garbage folder : 'images' --> 'test'")
    os.rename(PATH['garbage'], PATH['test'], )

    check = total_imgs == (garbage_df.shape[0] + len(os.listdir(PATH['test'])))
    if check:
        print(f"[INFO] Succesful processing")
        print(f"[INFO] {len(os.listdir(correct_mask_folder))} images in folder '{correct_mask_folder}' ")
        print(f"[INFO] {len(os.listdir(no_mask_folder))} images in folder '{no_mask_folder}' ")
        print(f"[INFO] {len(os.listdir(incorrect_mask_folder))} images in folder '{incorrect_mask_folder}' ")
        print(f"[INFO] {len(os.listdir(PATH['test']))} images in folder '{PATH['test']}'.")
    else:
        shutil.rmtree(PATH['IMAGES'])
        os.makedirs(PATH['IMAGES'], exist_ok=True)

        print(f"[INFO] Processing failed, re-run this function please.")


print(dataset())

images = []
size = 224, 224


def Get_dataset():
    data = []
    for root, _, file in tqdm(os.walk(PATH['IMAGES'])):
        for name in file:
            file_path = os.path.join(root, name)
            target = os.path.split(root)[-1]
            img = cv2.imread(os.path.join(root, name))
            im = cv2.resize(img, size)
            images.append(im)

            data += [{"file path": file_path, "target": target}]
    return pd.DataFrame(data)


global_data = Get_dataset()

#  CHECKING THE global data SHAPE
print(global_data.shape)

global_data.head(10)

test_df = pd.DataFrame.where(global_data[['file path']], global_data['target'] == 'test')
test_df = test_df.dropna()

print(test_df)

train_df = global_data.drop(test_df.index)

# to shuffle the data
train_df = train_df.sample(frac=1)
print(train_df)

# IMAGES VISUALIZATION

plt.figure(figsize=(12, 10))
for i in range(1, 5):
    img = cv2.imread(np.array(train_df['file path'][1:6])[i])
    ax = plt.subplot(2, 2, i)
    plt.imshow(img)
    plt.title(np.array(train_df['target'][1:6])[i])

plt.show()

# print unique classes
classes = train_df.target.unique().tolist()
print(classes)

train_df = train_df.sample(frac=1)
print(train_df)

# splitting the train images into train and validation images.

train_data, eval_data = train_test_split(train_df, test_size=0.3, random_state=42)
train_data = train_data.reset_index(drop=True)
eval_data = eval_data.reset_index(drop=True)

# DATA AUGMENTATION
EPOCHS = 30
BS = 32

# initialize an our data augmenter as an "empty" image data generator
train_datagen = ImageDataGenerator(rotation_range=20, rescale=1. / 255, shear_range=0.1, zoom_range=0.25,
                                   horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
eval_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_dataframe(train_data, x_col='file path', y_col='target', target_size=size,
                                                    classes=classes, class_mode='categorical', shuffle=True,
                                                    batch_size=BS, )
eval_generator = eval_datagen.flow_from_dataframe(eval_data, x_col='file path', y_col='target', target_size=size,
                                                  classes=classes, class_mode='categorical', shuffle=False,
                                                  batch_size=BS, )

# VISUALIZATION SOME TRANSFORMED IMAGES
plt.figure(figsize=(12, 12))
for i in range(0, 8):
    plt.subplot(2, 4, i + 1)
    for X_batch, Y_batch in train_generator:
        image = X_batch[3]
        plt.imshow(image)
        plt.title(Y_batch[3])
        break
plt.tight_layout()
plt.show()

# BUILDING MODEL WITH A PRETRAINED MODEL
INIT_LR = 1e-4

input_tensor = Input(shape=(224, 224, 3))
lastModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)

FirstModel = lastModel.output
FirstModel = AveragePooling2D(pool_size=(2, 2))(FirstModel)
FirstModel = Dropout(0.4)(FirstModel)
FirstModel = BatchNormalization()(FirstModel)
FirstModel = Flatten(name="flatten")(FirstModel)
FirstModel = Dense(1000, activation="relu")(FirstModel)
FirstModel = Dropout(0.5)(FirstModel)
FirstModel = Dense(3, activation="softmax")(FirstModel)

model = Model(inputs=lastModel.input, outputs=FirstModel)

for layer in lastModel.layers:
    layer.trainable = False

#  compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#  train the head of the network
print("[INFO] training head...")
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = eval_generator.n // eval_generator.batch_size

H = model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=eval_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=EPOCHS, verbose=2)
print(H)

# VISUALIZING MODEL PERFORMANCE
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']

loss = H.history['loss']
val_loss = H.history['val_loss']

epochs_range = range(EPOCHS)
plt.style.use("ggplot")
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("[INFO] saving mask detector model...")
model.save('mask_detector.h5')
