import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0'] 
y = pd.read_csv("labels.csv")["labels"]
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
noOfClasses = len(classes)
print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = tts(X, y, train_size=7500, test_size=2500, random_state=9)
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0

model = LogisticRegression(solver = 'saga', multi_class='multinomial').fit(x_train_scale, y_train)

prediction = model.predict(x_test_scale)
accuracy = accuracy_score(y_test, prediction)
print(accuracy)

capture = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = capture.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = grey.shape

        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))
        cv2.rectangle(grey, upper_left, bottom_right, (0, 255, 0), 2)
        region = grey[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
        pil = Image.fromarray(region)

        bwImage = pil.convert('L')
        bwImageResized = bwImage.resize((28, 28), Image.ANTIALIAS)
        imageInverted = PIL.ImageOps.invert(bwImageResized)

        pixelFilter = 20
        minpixel = np.percentile(imageInverted, pixelFilter)
        image_scaled = np.clip(imageInverted - minpixel, 0, 255)
        maxpixel = np.max(imageInverted)
        image_scaled = np.asarray(image_scaled)/maxpixel

        testsample = np.array(image_scaled).reshape(1, 784)
        testPrediction = model.predict(testsample)
        
        print('predicted class is: ', testPrediction)
        cv2.imshow('frame', grey)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

capture.release()
cv2.destroyAllWindows()