import os
import cv2
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
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# T = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY)

	I = cv2.warpAffine(img, W, (img.shape[1], img.shape[0]))
	
	# cv2.imshow("warped", im)
	# cv2.waitKey(0)
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

			# e = T[y, x] - I[y, x]
			e = I[y, x] - T[y, x]
			temp = temp + e*sd

	delta_p = np.matmul(H_inv, temp)

	# e = 0
	# for y in range(bb[0], bb[2], 1):
	# 	for x in range(bb[1], bb[3], 1):
	# 		e = e + T[y, x] - I[y, x]

	# print("e: ", e)
	# print("delta_p: ", np.linalg.norm(delta_p))

	return delta_p, I

def track_obj(img, T, T_bb, p, cnt, bb):
	# img_hist = cv2.cvtColor(img_hist, cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	T = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY)
	img_hist = deepcopy(img)
	img_hist = np.asarray(img_hist, dtype='uint8')
	crop_img = img_hist[bb[0]-10:bb[2]+10, bb[1]-10:bb[3]+10]
	img_hist = cv2.equalizeHist(crop_img)
	# cv2.imshow("cropped histogram", img_hist)
	# cv2.waitKey(0)
	# for i in range(bb[0],bb[2]):
	# 	for j in range(bb[1],bb[3]):
	# 		img[i][j] = img_hist[i][j]
	img[bb[0]-10:bb[2]+10, bb[1]-10:bb[3]+10] = img_hist
	# cv2.imshow("full image histogram", img)
	# cv2.waitKey(0)
	img = np.asarray(img, dtype='float32')
	T = np.asarray(T, dtype='float32')
	count = 0
	p_orig = p

	while count < 500:
		# img_copy = deepcopy(img)
		delta_p, I = traks_obj_util(img, T, p, T_bb)
		p = p + delta_p
		count = count + 1
		if np.linalg.norm(delta_p) < 0.05:
			print('Found at ', count)
			break
	if count == 500:
		print("Oh no!! Couldn't track object!!!! Tragedy!!!!!!")
		p = p_orig
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
	W = get_W(p)
	W_inv = cv2.invertAffineTransform(W)

	# bb = np.zeros((1, 4))
	T1 = np.asarray([T_bb[1], T_bb[0], 1])
	T2 = np.asarray([T_bb[3], T_bb[2], 1])

	x1, y1 = np.matmul(W_inv, np.reshape(T1, (3,1)))
	x2, y2 = np.matmul(W_inv, np.reshape(T2, (3,1)))
	return np.asarray([int(y1), int(x1), int(y2), int(x2)]), p


def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--test_video', type=int, default = 1, help = 'Choose the inut test video. (1: Car4, 2: Bolt2, 3: DragonBaby) Default value:1')

	Args = Parser.parse_args()
	test_video = Args.test_video

	if test_video == 1:
		rel_path = "Car4/img"
	elif test_video == 2:
		rel_path = "Bolt2/img"
	elif test_video == 3:
		rel_path = "DragonBaby/img"
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
	

	# T = images[0][51:138, 70:177, :]
	# cv2.imshow("template", T)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()
	T = images[160]
	T_bb = [61, 104, 125, 186]

	count = 0

	# initialize P
	p = np.asarray([0, 0, 0, 0, 0, 0])
	p = np.reshape(p, (6,1))
	bb = T_bb
	for i in range(161, len(images)):
		# if count == 0:
		# 	count = count + 1
		# 	continue
		print(bb)
		bb, p = track_obj(images[i], T, T_bb, p, i+1, bb)
		im = cv2.rectangle(images[i], (bb[1], bb[0]), (bb[3], bb[2]), color=(255,0,0), thickness=2)
		cv2.imwrite('Output/frame' + str(i+1) + '.png', im)
		# count = count + 1
		# if count == 5:
		# 	break

	# res = display_groundtruth(images, gt_path, test_video)

	# for im in res:
	# 	cv2.imshow('video', im)
	# 	if cv2.waitKey(0) & 0xff == 27:
	# 		cv2.destroyAllWindows()




if __name__ == "__main__":
	main()