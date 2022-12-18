
from matplotlib import pyplot as plt
import cv2
from tensorflow import keras
import numpy as np
import os
import json
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import LabelBinarizer


def fn(filename):
    img = cv2.imread(filename)

    img = cv2.resize(img, (img.shape[1], img.shape[0]))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(hsv)
    V = channels[2]
    _, thresh = cv2.threshold(V, 200, 255, cv2.THRESH_BINARY)
    vertical_hist = thresh.shape[0] - np.sum(thresh, axis=0, keepdims=True)/255
    flag = 0
    j = 10
    char_count = 0
    it = 0
    segment = np.zeros(
        (thresh.shape[0], (thresh.shape[1]//3) - 50), dtype=np.uint8)
    segment.fill(255)
    process_img = []
    for v in vertical_hist[0]:
        it += 1

        if(v != 0):
            flag = 1
            if(j < ((thresh.shape[1]//3) - 50)):
                segment[:, j] = thresh[:, it]
                j += 1
            continue

        if(flag == 1):
            char_count += 1
            process_img.append(segment)
            flag = 0
            j = 10
            segment = np.zeros(
                (thresh.shape[0], (thresh.shape[1]//3) - 50), dtype=np.uint8)
            segment.fill(255)

    return process_img


index_to_class = {0: 'ALPHA', 1: 'BETA', 2: 'CHI', 3: 'DELTA', 4: 'EPSILON', 5: 'ETA', 6: 'GAMMA', 7: 'IOTA', 8: 'KAPPA', 9: 'LAMDA', 10: 'MU', 11: 'NU',
                  12: 'OMEGA', 13: 'OMICRON', 14: 'PHI', 15: 'PI', 16: 'PSI', 17: 'RHO', 18: 'SIGMA', 19: 'TAU', 20: 'THETA', 21: 'UPSILON', 22: 'XI', 23: 'ZETA'}


model = keras.models.load_model('model')


def decaptcha(filenames):
   
#filenames = ['test/0.png', 'test/1.png']
    labels = []

    for filename in filenames:
        ret = fn(filename)
        imgs = []
        for img in ret:
            img = np.array(img)
            imgs.append(img)

        imgs = np.array(imgs)

        imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
        res = model.predict(imgs)
        pred = np.argmax(res, axis=1)
        temp = []
        for i in range(pred.shape[0]):
            temp.append(index_to_class[pred[i]])
        labels.append(",".join(temp))
    return labels


if __name__ == "__main__":
    filenames = ["test/0.png", "test/1.png"]
    labels = decaptcha(filenames)
    print(labels)


