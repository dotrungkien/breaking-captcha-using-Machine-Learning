import build_model
import requests
import cv2
import os
from os import listdir
from os.path import join, isfile
from prepare_data import reduce_noise, crop, adjust_folder
from build_model import predict_char

def get_captcha(limit=10):
	url = "https://chuyencuadev.com/captcha"
	for i in range (limit):
		filename = '{0:04}.jpg'.format(i)
		print(filename)
		with open(filename, 'wb') as f:
			response = requests.get(url)
			if response.ok: f.write(response.content)


def predict(filename):
	res = ''
	reduce_noise(filename)
	out_path = 'tmp/{0}/'.format(filename.split('.')[0])
	crop(filename, out_path)
	adjust_folder(out_path)
	for f in listdir(out_path):
		if isfile(join(out_path, f)) and 'jpg' in f:
			res += predict_char(out_path+f)
	print(res)
	return res
# get_captcha()
# for i in range(10):
# 	filename = '{0:04}.jpg'.format(i)
# 	predict(filename)

predict('0000.jpg')