from __future__ import division
from __future__ import print_function

import requests
import os
from os import listdir
from os.path import join, isfile
from PIL import Image, ImageChops
import math
import numpy as np
import cv2
import random
import string
from scipy.misc import imread

chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
chars_dict = {c: chars_list.index(c) for c in chars_list}

IMAGE_TOTAL = 1000
RAW_PATH = "data/raw/"
SLICED_PATH = "data/sliced/"

part = 0
list_chars = [f for f in listdir('data/chars') if isfile(join('data/chars', f)) and 'jpg' in f]

def crawl_images():
	url = "https://chuyencuadev.com/captcha"
	for i in range (1, IMAGE_TOTAL):
		file_path = join(RAW_PATH,'{0:04}.jpg'.format(i))
		print(file_path)
		with open(file_path, 'wb') as f:
			response = requests.get(url)
			if response.ok: f.write(response.content)

def process_directory(directory):
    file_list = []
    for file_name in listdir(directory):
        file_path = join(directory, file_name)
        if isfile(file_path) and 'jpg' in file_name:
            file_list.append(file_path)
    return file_list

def process_image(image_path):
    image = imread(image_path)
    image = image.reshape(1080,)
    return np.array([x/255. for x in image])

def reduce_noise(file_path):
	print(file_path)
	img = cv2.imread(file_path)
	dst = cv2.fastNlMeansDenoisingColored(img,None,50,50,7,21)
	cv2.imwrite(file_path, dst)
	img = Image.open(file_path).convert('L')
	img = img.point(lambda x: 0 if x<128 else 255, '1')
	img.save(file_path)

def reduce_noise_dir(directory):
	list_file = process_directory(directory)
	for file_path in list_file:
		print(file_path)
		img = cv2.imread(file_path)
		dst = cv2.fastNlMeansDenoisingColored(img,None,50,50,7,21)
		cv2.imwrite(file_path, dst)
		img = Image.open(file_path).convert('L')
		img = img.point(lambda x: 0 if x<128 else 255, '1')
		img.save(file_path)

def crop(file_path, out_directory):
	part = 0
	img = Image.open(file_path)
	p = img.convert('P')
	w, h = p.size

	letters = []
	left, right= -1, -1
	found = False
	for i in range(w):
		in_letter = False
		for j in range(h):
			if p.getpixel((i,j)) == 0:
				in_letter = True
				break
		if not found and in_letter:
			found = True
			left = i
		if found and not in_letter and i-left > 25:
			found = False
			right = i
			letters.append([left, right])
	origin = file_path.split('/')[-1].split('.')[0]
	for [l,r] in letters:
		if r-l < 40:
			bbox = (l, 0, r, h)
			crop = img.crop(bbox)
			crop = crop.resize((30,60))
			crop.save(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)))
			part += 1

def crop_dir(raw_directory, out_directory):
	list_file = process_directory(raw_directory)
	global part
	for file_path in list_file:
		print(file_path)
		img = Image.open(file_path)
		p = img.convert('P')
		w, h = p.size

		letters = []
		left, right= -1, -1
		found = False
		for i in range(w):
			in_letter = False
			for j in range(h):
				if p.getpixel((i,j)) == 0:
					in_letter = True
					break
			if not found and in_letter:
				found = True
				left = i
			if found and not in_letter and i-left > 25:
				found = False
				right = i
				letters.append([left, right])
		origin = file_path.split('/')[-1].split('.')[0]
		for [l,r] in letters:
			if r-l < 40:
				bbox = (l, 0, r, h)
				crop = img.crop(bbox)
				crop = crop.resize((30,60))
				crop.save(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)))
				part += 1

def adjust_dir(directory):
	list_file = process_directory(directory)
	for file_path in list_file:
		img = Image.open(file_path)
		p = img.convert('P')
		w, h = p.size
		start, end = -1, -1
		found = False
		for j in range(h):
			in_letter = False
			for i in range(w):
				if p.getpixel((i,j)) == 0:
					in_letter = True
					break
			if not found and in_letter:
				found = True
				start = j
			if found and not in_letter and j-start > 35:
				found = False
				end = j
		bbox = (0, start, w, end)
		crop = img.crop(bbox)
		crop = crop.resize((30,36))
		crop.save(file_path)

def rand_string(N=6):
	return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))

def rename(path, filename, letter):
	os.rename(join(path,filename), join(path, letter+'-' + rand_string() + '.jpg'))

def detect_char(path, filename):
	class Fit:
		letter = None
		difference = 0
	best = Fit()
	_img = Image.open(join(path, filename))
	for img_name in list_chars:
		current = Fit()
		img = Image.open(join('data/chars', img_name))
		current.letter = img_name.split('-')[0]
		difference = ImageChops.difference(_img, img)
		for x in range(difference.size[0]):
			for y in range(difference.size[1]):
				current.difference += difference.getpixel((x, y))/255.
		if not best.letter or best.difference > current.difference:
			best = current
	if best.letter == filename.split('-')[0]: return
	print(filename, best.letter)
	rename(path, filename, best.letter)

def detect_dir(directory):
	for f in listdir(directory):
		if isfile(join(directory, f)) and 'jpg' in f:
			detect_char(directory, f)

if __name__=='__main__':
	crawl_images()
	reduce_noise_dir(RAW_PATH)
	crop_dir(RAW_PATH, SLICED_PATH)
	adjust_dir(SLICED_PATH)
	pass
