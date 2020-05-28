from collections import OrderedDict
import numpy as np
import os
import cv2

# facial landmarks data

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])


FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS

# utility geometric functions

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype=np.float32):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=np.float32)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def shape_to_np2(shape, dtype=np.float32):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 1, 2), dtype=np.float32)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = ( shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


def getFlow(imPrev, imNew, landmarksPrev):

	lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS |
															 cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	next, status, error = cv2.calcOpticalFlowPyrLK(imPrev, imNew, landmarksPrev, None, **lk_params)
	# Selects good feature points for previous position
	good_old = landmarksPrev[status == 1]
	# Selects good feature points for next position
	good_new = next[status == 1]
	return good_old, good_new



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def getDenseFlow(imPrev, imNew):

	imPrev = cv2.cvtColor(imPrev, cv2.COLOR_BGR2GRAY)
	imNew = cv2.cvtColor(imNew, cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(imPrev, imNew, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	h, w = imPrev.shape[:2]
	step = 16
	y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
	fx, fy = flow[y, x].T
	lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv2.cvtColor(imPrev, cv2.COLOR_GRAY2BGR)
	cv2.polylines(vis, lines, 0, (0, 255, 0))
	for (x1, y1), (_x2, _y2) in lines:
		cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

	return vis

def get_denseFlow(imPrev, imNew):

	imPrev = cv2.cvtColor(imPrev, cv2.COLOR_BGR2GRAY)
	imNew = cv2.cvtColor(imNew, cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(imPrev, imNew, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	return flow

# function to build histogram
def calc_hist(flow):

	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=1)
	mag_list = []
	ang_list = []
	for sublist in mag:
		for item in sublist:
			mag_list.append(item)

	for sublist in ang:
		for item in sublist:
			ang_list.append(item)

	counter = 0

	q1, q2, q3, q4, q5, q6, q7, q8 = 0, 0, 0, 0, 0, 0, 0, 0
	for angg in ang_list:
		if ((0 < angg) & (angg <= 45)):
			q1 += mag_list[counter]
		elif ((45 < angg) & (angg <= 90)):
			q2 += mag_list[counter]
		elif ((90 < angg) & (angg <= 135)):
			q3 += mag_list[counter]
		elif ((135 < angg) & (angg <= 180)):
			q4 += mag_list[counter]
		elif ((180 < angg) & (angg <= 225)):
			q5 += mag_list[counter]
		elif ((225 < angg) & (angg <= 270)):
			q6 += mag_list[counter]
		elif ((270 < angg) & (angg <= 315)):
			q7 += mag_list[counter]
		elif ((315 < angg) & (angg <= 360)):
			q8 += mag_list[counter]

		counter += 1
	n_pixels = len(mag_list)
	#print(n_pixels)
	hist = [float(round(q1/n_pixels,2)), float(round(q2/n_pixels,2)),float(round(q3/n_pixels,2)),float(round(q4/n_pixels,2)),float(round(q5/n_pixels,2)),float(round(q6/n_pixels,2)),float(round(q7/n_pixels,2)),float(round(q8/n_pixels,2))]


	#print(hist)
	return (hist)


# functions to organize data in files

def write_test(hist_list, type):

	file = open("testData" + str(type) + ".txt", "w")
	file.write("+2 ")
	size = len(hist_list)
	i = 0
	counter = 1
	while i < size:
		y = 0
		while y < 8:
			hstring = str(hist_list[i][y])
			y += 1
			hstring = str(counter) + ":" + hstring + " "
			file.write(hstring)
			counter += 1
		i += 1
	file.write("\n")

	file.close()

	return


def prepare_data(hist_list, count):

	file = open("datax100_" + str(count) + ".txt", "a")
	file.write("+2 ")
	size = len(hist_list)
	i = 0
	counter = 1
	while i < size:
		y = 0
		while y < 8:
			hstring = str(hist_list[i][y])
			y += 1
			hstring = str(counter) + ":" + hstring + " "
			file.write(hstring)
			counter += 1
		i += 1
	file.write("\n")

	file.close()

	return


def endingWindow(count):
	file = open("datax100_" + str(count) + ".txt", "a")
	file.write("AAA end window")
	file.write("\n")

	return

def endingfolder(count):
	file = open("datax100_" + str(count) + ".txt", "a")
	file.write("END")
	file.write("\n")

	return

# get dense flow and store in a proper histogram

def calcAndStoreHOF(oldROIS, rois, hist_list0, hist_list1, hist_list2):
	imm1 = get_denseFlow(oldROIS[0], rois[0])
	imm2 = get_denseFlow(oldROIS[1], rois[1])
	imm3 = get_denseFlow(oldROIS[2], rois[2])
	hist1 = calc_hist(imm1)
	hist2 = calc_hist(imm2)
	hist3 = calc_hist(imm3)
	hist_list0.append(hist1)
	hist_list1.append(hist2)
	hist_list2.append(hist3)

	return
