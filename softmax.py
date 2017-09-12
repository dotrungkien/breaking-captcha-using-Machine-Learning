from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import requests
import cv2
import glob
import random
from prepare_data import process_directory, process_image, reduce_noise, crop, adjust_dir, rename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

learning_rate = 0.01
display_step = 1

model_path = "tmp/softmax.ckpt"
chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
chars_dict = {c: chars_list.index(c) for c in chars_list}
chars_index = {chars_list.index(key):key for key in chars_list}

def process_data(directory):
    images = []
    labels = []
    image_list = process_directory(directory)
    for image_path in image_list:
        images.append(process_image(image_path))
        label = chars_dict[image_path.split('/')[-1].split('-')[0]]
        labels.append([1 if i == label else 0 for i in range(36)])
    return np.array(images), np.array(labels)

def train_test_split():
    print("Loading images...")
    images, labels = process_data("data/chars/")
    n = len(images)
    n_train = int(n*0.8)
    images_train = images[:n_train]
    labels_train = labels[:n_train]
    images_test = images[n_train:]
    labels_test = labels[n_train:]
    print(images_train.shape, labels_train.shape, images_test.shape, labels_test.shape)
    return images_train, labels_train, images_test, labels_test

x = tf.placeholder(tf.float32, [None, 1080])
y = tf.placeholder(tf.float32, [None, 36])

W = tf.Variable(tf.zeros([1080, 36]))
b = tf.Variable(tf.zeros([36]))

pred = tf.nn.softmax(tf.matmul(x, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
def train():
    images_train, labels_train, images_test, labels_test = train_test_split()
    print("Start training...")
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            _, _cost = sess.run([optimizer, cost], feed_dict={x: images_train, y: labels_train})
        print("cost = {:.9f}".format(_cost))
        print("Optimization Finished!")

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: images_test, y: labels_test}))

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

def predict_char(img_path):
    img = np.array(Image.open(img_path).convert('L')).reshape(1080,)
    for i in range(1080): img[i] = img[i]/255.
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)
        predict = tf.argmax(pred, 1)
        char = chars_index[predict.eval({x:[img]})[0]]
        # print(char)
        return char

def predict(filename):
    res = ''
    reduce_noise(filename)
    out_path = 'tmp/null/'
    files = glob.glob(out_path+'*')
    for f in files: os.remove(f)
    crop(filename, out_path)
    adjust_dir(out_path)
    files = sorted([f for f in listdir(out_path) if isfile(join(out_path, f)) and 'jpg' in f])
    for f in files:
        res += predict_char(out_path+f)
    print(res)
    return res
# train()
predict_char("N-test_char.jpg")
predict("test_string.jpg")
