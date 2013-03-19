#!/usr/bin/python2

import cv
import cv2
import copy
import numpy
import math
import os
# import matplotlib.pyplot as plt # crashes when using opencv windows
import Gnuplot

gp = Gnuplot.Gnuplot()
gp("set style data lines")

gost_width = [520.0, 408.0]
gost_height = 112.0
gost_ratio = [gost_height / w for w in gost_width]
min_error = 0.15;

path = "../test_img/inet/"

img_index = 0
min_index = 0
max_img_index = len(os.listdir(path)) - 1
line_index = 0

def rms(values):
	rms = 0.0
	for v in values:
		rms += v**2;
	rms = math.sqrt(rms / len(values))

	return rms

def sma(values, size):
	new_values = []
	for i in range(len(values) - size):
		new_values.append(sum(values[i:i+size]) / float(size))
	return new_values

def compute():
	img_src = cv2.imread(path + "%03d.jpg" % img_index)

	img = cv2.cvtColor(img_src, cv.CV_RGB2GRAY)
	# img = cv2.blur(img, (3, 3))
	# # img = cv2.Canny(img, 400, 300)
	# # retval, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
	# img = cv2.adaptiveThreshold(img, 255,
	# 							cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 							cv2.THRESH_BINARY,
	# 							31, -10)
	# contours, _ = cv2.findContours(copy.copy(img),
	# 			  				   cv.CV_RETR_LIST,
	# 							   cv.CV_CHAIN_APPROX_SIMPLE)
	# for contour in contours:
	# 	rect = cv2.boundingRect(contour)
	# 	ratio = float(rect[3]) / rect[2]
	# 	if rect[3] * rect[2] > 800:
	# 		for gr in gost_ratio:
	# 			if abs(ratio - gr) < min_error:
	# 				cv2.rectangle(img_src,
	# 							  (rect[0], rect[1]),
	# 							  (rect[0] + rect[2], rect[1] + rect[3]),
	# 							  (255, 0, 0), 2)
	# 				break

	edge_matrix = numpy.array([[-0.0, -1.0, 0.0, 1.0, 0.0],
							   [-1.0, -2.0, 0.0, 2.0, 1.0],
							   [-0.0, -1.0, 0.0, 1.0, 0.0]])
	img = cv2.filter2D(img, -1, edge_matrix)

	py = []
	for y in range(img.shape[0]):
		py.append(rms(img[y, :]))

	pys = sma(py, 3)

	# pyp = []
	# for i in range(len(pys) - 1):
	# 	pyp.append(abs(pys[i + 1] - pys[i]))

	# pyps = sma(pyp, 1)

	line_index = pys.index(max(pys))

	
	cv2.line(img_src, (0, line_index), (img.shape[1] - 1, line_index), (255, 0, 0))
	cv2.imshow("img", img_src)
	cv2.imshow("img2", img)

	gp.plot(pys)


compute()

key = 0
while key != 27:
	key = cv2.waitKey()

	if key == 65363:
		img_index += 1
		if img_index > max_img_index:
			img_index = 0
		compute()
	elif key == 65361:
		img_index -= 1
		if img_index < 0:
			img_index = max_img_index
		compute()