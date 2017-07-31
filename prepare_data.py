from __future__ import division
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

part = 0
list_chars = [f for f in listdir('data/chars') if isfile(join('data/chars', f)) and 'jpg' in f]


def rand_string(N=6):
	return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))

def get_data():
	url = "https://chuyencuadev.com/captcha"
	for i in range (1, 1000):
		filename = '{0:04}.jpg'.format(i)
		print(filename)
		with open(filename, 'wb') as f:
			response = requests.get(url)
			if response.ok: f.write(response.content)

def reduce_noise(filename):
	img = cv2.imread(filename)
	dst = cv2.fastNlMeansDenoisingColored(img,None,50,50,7,21)
	cv2.imwrite(filename, dst)
	img = Image.open(filename).convert('L')
	img = img.point(lambda x: 0 if x<128 else 255, '1')
	img.save(filename)

def crop(filename, outpath):
	global part
	img = Image.open(filename)
	p = img.convert('P')
	w, h = p.size

	letters = []
	start, end = -1, -1
	found = False
	for i in range(w):
		in_letter = False
		for j in range(h):
			if p.getpixel((i,j)) == 0:
				in_letter = True
				break
		if not found and in_letter:
			found = True
			start = i
		if found and not in_letter and i-start > 25:
			found = False
			end = i
			letters.append([start, end])
	origin = filename.split('/')[-1].split('.')[0]
	for [l,r] in letters:
		if r-l < 40:
			bbox = (l, 0, r, h)
			crop = img.crop(bbox)
			crop = crop.resize((30,60))
			crop.save(outpath + '{0:04}_{1}.jpg'.format(part, origin))
			part += 1
			
def adjust(path, filename):
	img = Image.open(join(path, filename))
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
	crop.save(join(path, filename))

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

def adjust_folder(path):
	for f in listdir(path):
		if isfile(join(path, f)) and 'jpg' in f:
			adjust(path, f)
def detect_folder(path):
	for f in listdir(path):
		if isfile(join(path, f)) and 'jpg' in f:
			detect_char(path, f)

if __name__=='__main__':
	# for i in range(1, 800):
	# 	filename = 'data/train/{0:04}.jpg'.format(i)
	# 	print(filename)
	# 	crop(filename, 'data/train/sliced/')
	# for i in range(800, 1000):
	# 	filename = 'data/test/{0:04}.jpg'.format(i)
	# 	print(filename)
	# 	crop(filename, 'data/test/sliced/')
	# adjust_folder('data/chars/')
	# adjust_folder('data/train/sliced')
	# adjust_folder('data/test/sliced')
	# detect_folder('data/train/sliced')
	reduce_noise('1.jpg')
	crop('1.jpg', 'viblo/')
	adjust('viblo/', '0000_1.jpg')
	adjust('viblo/', '0001_1.jpg')
	adjust('viblo/', '0002_1.jpg')
	adjust('viblo/', '0003_1.jpg')
	adjust('viblo/', '0004_1.jpg')
	adjust('viblo/', '0005_1.jpg')
	pass