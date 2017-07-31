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
from prepare_data import reduce_noise, crop, adjust_folder, rename, detect_folder


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Parameters
learning_rate = 0.01
training_epochs = 25
display_step = 1
k = 36
model_path = "tmp/model.ckpt"
chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

d = {}
for c in chars_list:
    vector = [0]*k
    vector[chars_list.index(c)] = 1
    d[c] = vector
chars = {chars_list.index(key):key for key in chars_list}

def img_preprocessing():
    images = []
    labels = []
    all_files = [f for f in listdir('data/chars/') if isfile(join('data/chars/', f)) and 'jpg' in f]
    for name in all_files:
        labels.append(d[name.split('-')[0]])
        img_path = 'data/chars/' + name
        img = np.array(Image.open(img_path).convert('L')).reshape(1080,)
        for i in range(1080):
            img[i] = img[i]/255.
        images.append(img)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

images, labels = img_preprocessing()
n = len(images)
n_train = int(n*0.8)
images_train = images[:n_train]
labels_train = labels[:n_train]
images_test = images[n_train:]
labels_test = labels[n_train:]

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 1080])
y = tf.placeholder(tf.float32, [None, 36])

# Set model weights
W = tf.Variable(tf.zeros([1080, 36]))
b = tf.Variable(tf.zeros([36]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
def train():
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = 1000
            # Loop over all batches
            for i in range(total_batch):
                _, c = sess.run([optimizer, cost], feed_dict={x: images_train,
                                                              y: labels_train})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: images_test, y: labels_test}))

        #save model
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

def predict_char(img_path):
    img = np.array(Image.open(img_path).convert('L')).reshape(1080,)
    for i in range(1080): img[i] = img[i]/255.
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)
        predict = tf.argmax(pred, 1)
        return chars[predict.eval({x:[img]})[0]]

def predict(filename):
    res = ''
    reduce_noise(filename)
    out_path = 'tmp/null/'
    files = glob.glob(out_path+'*')
    for f in files: os.remove(f)
    crop(filename, out_path)
    adjust_folder(out_path)
    for f in listdir(out_path):
        if isfile(join(out_path, f)) and 'jpg' in f:
            res += predict_char(out_path+f)
    print(res)
    return res

fs = [f for f in listdir('data/predict/') if isfile(join('data/predict', f)) and 'jpg' in f]
total = len(fs)
for f in fs:
    char = predict_char('data/predict/'+f)
    if char != f.split('-')[0]: 
        rename('data/predict/', f, char)
# detect_folder('data/predict/')