from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prepare_data import process_directory, process_image, reduce_noise, crop, adjust_dir, rename


chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
chars_dict = {c: chars_list.index(c) for c in chars_list}

def process_data(directory):
    images = []
    labels = []
    image_list = process_directory(directory)
    for image_path in image_list:
        images.append(process_image(image_path))
        labels.append(chars_dict[image_path.split('/')[-1].split('-')[0]])
    return np.array(images), np.array(labels).reshape([len(labels), 1])

def train():
    print("Loading images....")
    images, labels = process_data("data/chars/")
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.2, random_state=42)
    print("Training...")
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(images_train)
    dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=36, 
    																				feature_columns=feature_columns)
    dnn_clf.fit(x=images_train, y=labels_train, batch_size=50, steps=40000)
    labels_pred = list(dnn_clf.predict(images_test))
    print(accuracy_score(labels_test, labels_pred))

if __name__ == '__main__':
	train()