import os
import cv2
import math
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def display_groundtruth(images, gt_path, test_video):
	f = open(gt_path, "r")
	result = []

	for im in images:
		bb = f.readline()
		if test_video == 1:
			bb = bb.split()
		else:
			bb = bb.split(',')
		bb = np.asarray(bb, dtype=int)
		im = cv2.rectangle(im, (bb[0], bb[1]), (bb[2]+bb[0], bb[3]+bb[1]), color=(255,0,0), thickness=2)
		result.append(im)

	result = np.asarray(result)

	return result

def get_W(p):
	W = np.asarray([[p[0][0]+1, p[2][0], p[4][0]], 
				    [p[1][0], p[3][0]+1, p[5][0]]], dtype = 'float32')

	return W

def get_Winv(p):
	W_inv = np.zeros((2,3))

	W_inv[0][0] = p[1][0]
	W_inv[0][1] = -(1 + p[0][0])
	W_inv[0][2] = (1 + p[0][0])*p[5][0] - p[1][0]*p[4][0]
	W_inv[1][0] = -(1 + p[3][0])
	W_inv[1][1] = p[2][0]
	W_inv[1][2] = (1 + p[3][0])*p[4][0] - p[2][0]*p[5][0]


	return W_inv

def traks_obj_util(img, T, p, bb):
	W = get_W(p)
	W = cv2.invertAffineTransform(W)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	T = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY)

	I = cv2.warpAffine(img, W, (img.shape[1], img.shape[0]))
	
	
	# cv2.imshow("warped", I)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()
	# cv2.imshow("template", T)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

	grad_x = cv2.warpAffine(grad_x, W, (grad_x.shape[1], grad_x.shape[0]))
	grad_y = cv2.warpAffine(grad_y, W, (grad_y.shape[1], grad_y.shape[0]))

	H = np.zeros((6,6))
	for y in range(bb[0], bb[2], 1):
		for x in range(bb[1], bb[3], 1):
			del_W = np.asarray([[x, 0, y, 0, 1, 0],
							    [0, x, 0, y, 0, 1]])
			grad = np.asarray([grad_x[y,x], grad_y[y,x]])
			# grad = [grad_x[y,x], grad_y[y,x]]
			# print(grad.shape)
			# print(del_W.shape)

			sd = np.matmul(grad, del_W)
			sd = np.reshape(sd, (1,6))
			H_ = np.matmul(sd.transpose(), sd)
			H = H + H_

	H_inv = np.linalg.inv(H) 

	temp = np.zeros((6,1))
	for y in range(bb[0], bb[2], 1):
		for x in range(bb[1], bb[3], 1):
			del_W = np.asarray([[x, 0, y, 0, 1, 0],
							    [0, x, 0, y, 0, 1]])
			# grad = [grad_x[y,x], grad_y[y,x]]
			grad = np.asarray([grad_x[y,x], grad_y[y,x]])

			sd = np.matmul(grad, del_W)
			sd = np.reshape(sd, (1,6)).T

			e = T[y, x] - I[y, x]
			# e = I[y, x] - T[y, x]
			temp = temp + e*sd

	delta_p = np.matmul(H_inv, temp)

	e = 0
	for y in range(bb[0], bb[2], 1):
		for x in range(bb[1], bb[3], 1):
			# e = e + T[y, x] - I[y, x]
			e = e + (I[y, x] - T[y, x])**2

	# print("e: ", math.sqrt(e))
	# print("delta_p: ", delta_p)

	return delta_p, I, math.sqrt(e)

def track_obj(img, T, T_bb, p, cnt):

	img = np.asarray(img, dtype='float32')
	T = np.asarray(T, dtype='float32')

	count = 0

	while count < 500:
		# img_copy = deepcopy(img)
		delta_p, I, e = traks_obj_util(img, T, p, T_bb)
		p = p + delta_p
		count = count + 1
		if np.linalg.norm(delta_p) < 0.01:
			break

		# W_inv1 = get_Winv(p)
		# print(W_inv1)
		# W = get_W(p)
		# W_inv = cv2.invertAffineTransform(W)
		# # print(W_inv)

		# T1 = np.asarray([T_bb[1], T_bb[0], 1])
		# T2 = np.asarray([T_bb[3], T_bb[2], 1])

		# x1, y1 = np.matmul(W_inv, np.reshape(T1, (3,1)))
		# x2, y2 = np.matmul(W_inv, np.reshape(T2, (3,1)))

		# print(x1, y1, x2, y2)

		# im = cv2.rectangle(img_copy, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
		# cv2.imshow("image", img_copy)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

	cv2.imwrite('Warped/frame' + str(cnt) + '.png', I)

	print(np.linalg.norm(delta_p))
	print("e: ", e)
	W_inv = get_W(p)
	

	# bb = np.zeros((1, 4))
	T1 = np.asarray([T_bb[1], T_bb[0], 1])
	T2 = np.asarray([T_bb[3], T_bb[2], 1])

	x1, y1 = np.matmul(W_inv, np.reshape(T1, (3,1)))
	x2, y2 = np.matmul(W_inv, np.reshape(T2, (3,1)))

	return np.asarray([x1, y1, x2, y2]), p


def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--test_video', type=int, default = 1, help = 'Choose the inut test video. (1: Car4, 2: Bolt2, 3: DragonBaby) Default value:1')

	Args = Parser.parse_args()
	test_video = Args.test_video

	if test_video == 1:
		rel_path = "Car4/img"
		T_bb = [51, 70, 138, 177]
	elif test_video == 2:
		rel_path = "Bolt2/img"
		T_bb = [75, 269, 139, 303]
	elif test_video == 3:
		rel_path = "DragonBaby/img"
		T_bb = [83, 160, 148, 216]
	else:
		print("Input should be between 1-3. Can you count??")
		exit()

	cur_path = os.path.dirname(os.path.abspath(__file__))
	img_path = os.path.join(cur_path, rel_path)
	gt_path = os.path.join(img_path, '../groundtruth_rect.txt')

	images = []
	
	for name in sorted(os.listdir(img_path)):
		im = cv2.imread(os.path.join(img_path, name))
		images.append(im)

	images = np.asarray(images)
	
	# print(len(images))

	# T = images[0][51:138, 70:177, :]
	# cv2.imshow("template", T)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()
	T = images[0]
	# T_bb = [51, 70, 138, 177]

	count = 0

	# initialize P
	p = np.asarray([0, 0, 0, 0, 0, 0])
	p = np.reshape(p, (6,1))

	for i in range(len(images)):
		if count == 0:
			count = count + 1
			continue
		bb, p = track_obj(images[i], T, T_bb, p, count)
		im = cv2.rectangle(images[i], (bb[0], bb[1]), (bb[2], bb[3]), color=(255,0,0), thickness=2)
		cv2.imwrite('Output/frame' + str(i) + '.png', im)
		count = count + 1
		# if count == 5:
			# break

	# res = display_groundtruth(images, gt_path, test_video)

	# for im in res:
	# 	cv2.imshow('video', im)
	# 	if cv2.waitKey(0) & 0xff == 27:
	# 		cv2.destroyAllWindows()




if __name__ == "__main__":
	main()