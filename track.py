import os
import cv2
import math
import argparse
import numpy as np
from copy import deepcopy
import sys


def shift_data(tmp_mean, img):
	tmp_mean_matrix = np.full((img.shape), tmp_mean)
	# print(tmp_mean)
	# print(find_avg_brightness(img))
	img_mean_matrix = np.full((img.shape), find_avg_brightness(img))
	std_ = np.std(img)
	z_score = np.true_divide((img.astype(int) - tmp_mean_matrix.astype(int)), std_)
	dmean = np.mean(img) - tmp_mean
	
	if dmean < 10:
		shifted_img = -(z_score * std_).astype(int) + img_mean_matrix.astype(int)
	
	else:
		shifted_img = (z_score * std_).astype(int) + img_mean_matrix.astype(int)
	
	return shifted_img.astype(dtype=np.uint8)

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
		im = cv2.rectangle(im, (bb[0], bb[1]), (bb[2]+bb[0], bb[3]+bb[1]), color=(0,0,255), thickness=2)
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
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	T = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY)
	W = cv2.invertAffineTransform(W)

	I = cv2.warpAffine(img, W, (img.shape[1], img.shape[0]))

	grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

	grad_x = cv2.warpAffine(grad_x, W, (grad_x.shape[1], grad_x.shape[0]))
	grad_y = cv2.warpAffine(grad_y, W, (grad_y.shape[1], grad_y.shape[0]))


	H = np.zeros((6,6))
	temp = np.zeros((6,1))

	error = T - I
	err = np.reshape(error,(-1,1))
	sigma = np.std(err)


	for y in range(bb[0], bb[2], 1):
		for x in range(bb[1], bb[3], 1):
			del_W = np.asarray([[x, 0, y, 0, 1, 0],
							    [0, x, 0, y, 0, 1]])
			grad = np.asarray([grad_x[y,x], grad_y[y,x]])
		
			sd = np.matmul(grad, del_W)
			sd = np.reshape(sd, (1,6))

			e = T[y, x] - I[y, x]

			# t=e**2                                                  #Implementation of huber function

			# if 0<= t and t <= sigma**2:                             
			# 	rho = 0.5*t

			# elif t>sigma**2: 
			# 	rho = sigma*math.sqrt(t) - 0.5*(sigma**2)
		
			# H_ = rho*np.matmul(sd.transpose(), sd)
			# temp =  temp + rho*sd.transpose()*e
			H_ = np.matmul(sd.transpose(), sd)
			temp =  temp + sd.transpose()*e
			H = H + H_

	H_inv = np.linalg.inv(H) 
	delta_p = np.matmul(H_inv, temp)

	e = 0
	for y in range(bb[0], bb[2], 1):
		for x in range(bb[1], bb[3], 1):
			e = e + (I[y, x] - T[y, x])**2

	return delta_p, I, math.sqrt(e)

def find_avg_brightness(img):
	img_brightness = []
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img_brightness.append(0.2126*img[i][j][2] + 0.7152 * img[i][j][1] + 0.0722 * img[i][j][0])
	return np.mean(img_brightness)


def track_obj_dragon(img, T_front, T_bb_front, T_side, T_bb_side, T_back, T_bb_back, p_front, p_side, p_back, cnt, bb):
	img = np.asarray(img, dtype='float32')
	T_front = np.asarray(T_front, dtype='float32')
	T_side = np.asarray(T_side, dtype='float32')
	T_back = np.asarray(T_back, dtype='float32')
	count = 0
	p_orig_front = p_front
	p_orig_side = p_side
	p_orig_back = p_back
	p_front = np.asarray([0, 0, 0, 0, 0, 0])
	p_front = np.reshape(p_front, (6, 1))
	p_side = np.asarray([0, 0, 0, 0, 0, 0])
	p_side = np.reshape(p_side, (6, 1))
	p_back = np.asarray([0, 0, 0, 0, 0, 0])
	p_back = np.reshape(p_back, (6, 1))
	found = True
	front_p = 0
	front_dp = 0
	front_I = 0
	side_p = 0
	side_dp = 0
	side_I = 0
	back_p = 0
	back_dp = 0
	back_I = 0
	min_error_front = sys.maxsize
	min_error_side = sys.maxsize
	min_error_back = sys.maxsize
	print("Check for Front")
	while count < 300:
		delta_p, I, e = traks_obj_util(img, T_front, p_front, T_bb_front)
		p_front = p_front + delta_p
		count = count + 1
		if e < min_error_front:
			min_error_front = e
			front_dp = delta_p
			front_p = p_front
			front_I = I
		# if e < 2500:
		# 	print('Found front at ', count)
		# 	T1 = np.asarray([T_bb_front[1], T_bb_front[0], 1])
		# 	T2 = np.asarray([T_bb_front[3], T_bb_front[2], 1])
		# 	break
	# if count == 500:
	print("front e ", min_error_front)
	count = 0
	print("Check for Side")
	# p = p_orig_side
	while count < 300:
		delta_p, I, e = traks_obj_util(img, T_side, p_side, T_bb_side)
		p_side = p_side + delta_p
		count = count + 1
		if e < min_error_side:
			min_error_side = e
			side_dp = delta_p
			side_I = I
			side_p = p_side
			# print('Found side at ', count)
			# T1 = np.asarray([T_bb_side[1], T_bb_side[0], 1])
			# T2 = np.asarray([T_bb_side[3], T_bb_side[2], 1])
			# break
	# if count == 500:
	print("side e ", min_error_side)
	
	count = 0
	print("Check for Back")
	# p = p_orig_side
	while count < 300:
		delta_p, I, e = traks_obj_util(img, T_back, p_back, T_bb_back)
		p_back = p_back + delta_p
		count = count + 1
		if e < min_error_back:
			min_error_back = e
			back_dp = delta_p
			back_I = I
			back_p = p_back
	print("back e ", min_error_back)
	# count = 0
	# print("Check for Back")
	# p = p_orig
	# while count < 300:
	# 	delta_p, I, e = traks_obj_util(img, T_back, p, T_bb_back)
	# 	p = p + delta_p
	# 	count = count + 1
	# 	if e < min_error_back:
	# 		min_error_back = e
	# 		back_dp = delta_p
	# 		back_I = I
	# 		back_p = p
	# 		# print('Found back at ', count)
	# 		# T1 = np.asarray([T_bb_back[1], T_bb_back[0], 1])
	# 		# T2 = np.asarray([T_bb_back[3], T_bb_back[2], 1])
	# 		# break
	# print("back e ", min_error_back)
	if min_error_front < min_error_side and min_error_front < min_error_back:
		delta_p = front_dp
		I = front_I
		p = front_p
		T1 = np.asarray([T_bb_front[1], T_bb_front[0], 1])
		T2 = np.asarray([T_bb_front[3], T_bb_front[2], 1])
		e = min_error_front
		p_side = p_orig_side
		p_back = p_orig_back
		print("Front selected")
		
	elif min_error_side < min_error_front and min_error_side < min_error_back:
		delta_p = side_dp
		I = side_I
		p = side_p
		T1 = np.asarray([T_bb_side[1], T_bb_side[0], 1])
		T2 = np.asarray([T_bb_side[3], T_bb_side[2], 1])
		e = min_error_side
		p_front = p_orig_front
		p_back = p_orig_back
		print("Side selected")
		
	else:
		delta_p = back_dp
		I = back_I
		p = back_p
		T1 = np.asarray([T_bb_back[1], T_bb_back[0], 1])
		T2 = np.asarray([T_bb_back[3], T_bb_back[2], 1])
		e = min_error_back
		p_side = p_orig_side
		p_front = p_orig_front
		print("Back selected")
	
	if e > 3000:
		print("Oh no, couldn't track!")
		if min_error_front < min_error_side and min_error_front < min_error_back:
			p = p_orig_front
		elif min_error_side < min_error_back and min_error_side < min_error_front:
			p = p_orig_side
		else:
			p = p_orig_back
		found = False
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
	
	# cv2.imwrite('Warped_dragon/frame' + str(cnt) + '.png', I)
	
	# print(np.linalg.norm(delta_p))
	# print("e: ", e)
	W_inv = get_W(p)
	if found:
		x1, y1 = np.matmul(W_inv, np.reshape(T1, (3, 1)))
		x2, y2 = np.matmul(W_inv, np.reshape(T2, (3, 1)))
	else:
		y1 = bb[0]
		x1 = bb[1]
		y2 = bb[2]
		x2 = bb[3]
	# bb = np.zeros((1, 4))
	
	return np.asarray([int(y1), int(x1), int(y2), int(x2)]), p_front, p_side, p_back


def track_obj_car(img, T, T_bb, p, cnt, bb):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	T = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY)
	# img_hist = deepcopy(img)                                               # Implementation of Histogram equilization
	# img_hist = np.asarray(img_hist, dtype='uint8')
	# crop_img = img_hist[bb[0]-10:bb[2]+10, bb[1]-10:bb[3]+10]
	# img_hist = cv2.equalizeHist(crop_img)

	# img[bb[0]-10:bb[2]+10, bb[1]-10:bb[3]+10] = img_hist

	img = np.asarray(img, dtype='float32')
	T = np.asarray(T, dtype='float32')
	count = 0
	p_orig = p

	while count < 500:
		# img_copy = deepcopy(img)
		delta_p, I, e = traks_obj_util(img, T, p, T_bb)
		p = p + delta_p
		count = count + 1
		if np.linalg.norm(delta_p) < 0.001:
			print('Found at ', count)
			break
		if np.linalg.norm(p) >= 185:
			p = p_orig
			break
	if count == 500:
		print("Oh no!! Couldn't track object!!!! Tragedy!!!!!!")
		p = p_orig

	# cv2.imwrite('Warped_car/frame' + str(cnt) + '.png', I)

	print("norm of delta p", np.linalg.norm(delta_p))
	print("e: ", e)
	W_inv = get_W(p)
	
	T1 = np.asarray([T_bb[1], T_bb[0], 1])
	T2 = np.asarray([T_bb[3], T_bb[2], 1])

	x1, y1 = np.matmul(W_inv, np.reshape(T1, (3,1)))
	x2, y2 = np.matmul(W_inv, np.reshape(T2, (3,1)))
	return np.asarray([int(y1), int(x1), int(y2), int(x2)]), p

def generate_video(gt_path,test_video,output_path,vid_path):
	cur_path = os.path.dirname(os.path.abspath(__file__))
	img_path = os.path.join(cur_path,output_path )
	images = []
	
	for name in sorted(os.listdir(img_path)):
		# print(name)
		im = cv2.imread(os.path.join(img_path, name))
		images.append(im)

	images = np.asarray(images)

	res = display_groundtruth(images, gt_path, test_video)

	vidWriter = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (res[0].shape[1], res[0].shape[0]))
	# count = 1
	for img in res:
		vidWriter.write(img)
		# print(count)
		# count= count+1
		# if count ==5:
		# 	break
	vidWriter.release()

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--test_video', type=int, default = 3, help = 'Choose the inut test video. (1: Car4, 2: Bolt2, 3: DragonBaby) Default value:1')

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
	
	if test_video ==3:

		T_front = images[0]
		T_bb_front = [83, 160, 148, 216]
		T_side = images[10]
		T_bb_side = [59, 145, 130, 210]
		T_back = images[27]
		T_bb_back = [80, 190, 145, 255]


		T = images[0]

		count = 0
		if not os.path.exists('Output_dragon'):
			os.makedirs('Output_dragon')

		# initialize P
		p_front = np.asarray([0, 0, 0, 0, 0, 0])
		p_front = np.reshape(p_front, (6,1))
		p_side = np.asarray([0, 0, 0, 0, 0, 0])
		p_side = np.reshape(p_side, (6,1))
		p_back = np.asarray([0, 0, 0, 0, 0, 0])
		p_back = np.reshape(p_back, (6,1))
		bb = T_bb_front
		# temp_avg = find_avg_brightness(T[T_bb[0] - 10:T_bb[2] + 10, T_bb[1] - 10:T_bb[3] + 10, :])
		for i in range(len(images)):
			# if count == 0:
			# 	count = count + 1
			# 	continue
			img = deepcopy(images[i])
			bb, p_front, p_side, p_back = track_obj_dragon(images[i], T_front, T_bb_front, T_side, T_bb_side, T_back, T_bb_back, p_front, p_side, p_back, count, bb)
			img = cv2.rectangle(img, (bb[1], bb[0]), (bb[3], bb[2]), color=(255,0,0), thickness=2)

			cv2.imwrite('Output_dragon/frame%03d.png' % count, img)
			count = count + 1

		output_path =  "Output_dragon/"
		vid_path= "./DragonBabyResult.mp4"
		generate_video(gt_path,test_video, output_path, vid_path)


	if test_video ==1:

		p = np.asarray([0, 0, 0, 0, 0, 0])
		p = np.reshape(p, (6,1))
		bb = T_bb
		count ==0
		if not os.path.exists('Output_car'):
			os.makedirs('Output_car')
		# if os.path.exists("p_values.txt"):
		# 	os.remove("p_values.txt")
		# 	f = open("p_values.txt", "a")
		for i in range(len(images)):
			# if count == 0:
			# 	count = count + 1
			# 	continue
			# np.savetxt(f, p , fmt="%s", newline=' ')
			# f.write("\n")
			print("norm of p",np.linalg.norm(p))
			bb, p = track_obj(images[i], T, T_bb, p, i+1, bb)
			img = cv2.rectangle(images[i], (bb[1], bb[0]), (bb[3], bb[2]), color=(255,0,0), thickness=2)
			
			cv2.imwrite('Output_car/frame%03d.png' % count, img)
			count = count + 1

			print("```````````````````````````````````````", str(i+1))

		output_path =  "Output_car/"
		vid_path= "./CarResult.mp4"
		generate_video(gt_path,test_video, output_path, vid_path)




if __name__ == "__main__":
	main()