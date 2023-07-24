<h2 align="center"> FACE MASK DETECTION SYSTEM  </h2>
<br><br>



## 😷 FACE MASK DETECTION  😷

A face mask detection is a system capable of detecting if one or more individuals is/ are wearing their face mask. This project displays the number of faces detected, the number of peaople wearing the face mask **correctly**, **incorrectly** or **not** in a picture. 
With the increasing number of COVID-19 victims, integrating systems capable of detecting people wearing a face mask properly or not will help us track them easily. I please everyone to wear their face mask correctly to reduce the risk of contaminating one another.
<br><br>


## ⚠️ TECHONLOGY USED

* [OPENCV](https://opencv.org/about/)

* [HAAR-CASCADE CLASSIFIER](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

* [TENSORFLOW](https://www.tensorflow.org/)

* [FLASK](https://en.wikipedia.org/wiki/Flask_(web_framework))

<br><br>

## ⚙️ HOW THE SYSTEM WORKS?

This system is a mask detection project that detects face mask on people'face. This flask web app detects if someone is wearing a face mask correctly, incorrectly or not.
This system retrieves an image uploaded extract the faces detected in the image, feed each face into a serialized model `mask_detector.h5` which outputs a tuple composed of 3 values. The first value indicates the probability of no mask, the second value is the probability of a correct mask and the last one is the probability of the incorrect mask.
In the output picture, ach face is bordered by the color indicating his situation (No mask, Correct, Incorrect).
This website is built from scratch with HTML and CSS deployed to Flask use yourlaptop to test.

**DATA COLLECTION**: To Download the dataset check this [repository](https://github.com/cabani/MaskedFace-Net)


## 🔗 PROJECT FILES

The system depends on the following files.

1. `app.py` [See here](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/app.py): The flask app
2. `faceMask_model.py` [See here](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/faceMask_model.py): This file is the model used to build this project
3. `masks_detector.py` [See here](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/masks_detector.py): This file is capable of detecting face mask in a live real-time video.
4. `mask_detector_image.py` [See here](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/mask_detector_image.py) :This python file detects face mask from an image uploaded from the system
5. `mask_detector_video.py` [See here](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/mask_detector_video.py):This python file detects face mask from an video uploaded from the system
6. `haarcascade_frontalface_default.xml` [See here](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/haarcascade_frontalface_default.xml) :The haar cascade classifier used for face detection.
7. `model_detector.h5` [see here](https://bitbucket.org/memudu_alimatou/face-mask-detection-web-app-flask-python/src): this file is a serialized pickle file which accpets an image or a video and output a tuple of probabilities determining if the face detected in the inputted file has a face mask wore properly or not. 
8. `multi-face mask5B.ipynb` [See here](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/faceMask_model.ipynb) :this file is the model used to build this project check it.
 <br><br>

## ⌛ Project Demo

<br>

![capture](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/ezgif.com-gif-maker%20(3).gif)<br>
<br><br>

## 🔑 PEREQUISITES

All the dependencies and required libraries are included in the file **requirements.txt** [See here](https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON/blob/main/requirements.txt)


## 🚀 INSTALLATION

Clone the repo\
```$ git clone https://github.com/memudualimatou/FACE-MASK-DETECTION-WEB-APP-FLASK-PYTHON.git```


Change your directory to the cloned repo and create a Python virtual environment named 'test'

```$ mkvirtualenv test```


Now, run the following command in your Terminal/Command Prompt to install the libraries required

```$ pip3 install -r requirements.txt```


To download the **mask_detector.h5 (The serialized model of this project)** File [click here](https://bitbucket.org/memudu_alimatou/face-mask-detection-web-app-flask-python/src), you can clone my bitbucket repository to download the file or run 
`faceMask_model.py` on your local environment to save it. This file is too large to be uploaded here.

## 👏 And it's done!
Feel free to mail me for any doubts/query ✉️ alimatousadia005@gmail.com

##  🤝 Contribution
Feel free to file a new issue with a respective title and description on the this mask detection repository. If you already found a solution to your problem, I would love to review your pull request!

## ❤️ Owner
Made with ❤️  by MEMUDU Alimatou Sadia Anike
