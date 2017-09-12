from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import join, isfile
from scipy.misc import imread
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from prepare_data import reduce_noise, crop, adjust_folder, rename, detect_folder
import glob

chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
chars_dict = {c: chars_list.index(c) for c in chars_list}

def process_directory(directory):
    file_list = []

    for file_name in listdir(directory):
        file_path = join(directory, file_name)
        if isfile(file_path) and 'jpg' in file_name:
            file_list.append(file_path)
    return file_list

def process_directory2(directory):
    file_list = []

    for root, _, files in os.walk(directory):
        print(root, files)
        for file_name in files:
            file_path = join(root, file_name)
            if isfile(file_path) and 'jpg' in file_path:
                file_list.append(file_path)
    return file_list

def process_image(image_path):
    image = imread(image_path)
    image = image.reshape(1080,)
    return np.array([x/255. for x in image])

def process_data(directory):
    images = []
    labels = []
    image_list = process_directory(directory)
    for image_path in image_list:
        images.append(process_image(image_path))
        labels.append(chars_dict[image_path.split('/')[-1].split('-')[0]])
    return np.array(images), np.array(labels)

def train():
    print("Loading images....")
    images, labels = process_data("data/chars/")
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.2, random_state=42)
    print("Training...")
    clf = SVC(kernel="linear", C=10e5)
    # scores = cross_val_score(clf, images, labels, cv = 10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf.fit(images_train, labels_train)
    print(clf.score(images_test, labels_test))
    joblib.dump(clf, "svm.pkl")
    print("Saved model to svm.pkl")

def predict_char(image_path):
    image = process_image(image_path).reshape(1, -1)
    clf = joblib.load("svm.pkl")
    name = image_path.split('/')[-1].split('-')[0]
    actual = chars_list[clf.predict(image)[0]]
    # print("Predicted: {0}. Actual: {1}".format(name, actual))
    return actual

def predict_string(file_path):
    res = ''
    reduce_noise(file_path)
    out_path = 'tmp/null/'
    files = glob.glob(out_path+'*')
    for f in files: os.remove(f)
    crop(file_path, out_path)
    adjust_folder(out_path)
    file_list = process_directory(out_path)
    for f in sorted(file_list):
        res += predict_char(f)
    print(res)
    return res

if __name__=='__main__':
    # train()
    # predict_char("N-test_char.jpg")
    predict_string("test_string.jpg")
    pass
