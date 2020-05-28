# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

def getROI(image):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    ROIS = []
    # load the input image, resize it, and convert it to grayscale


    image = imutils.resize(image, width=500)
    gray = image

    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            if name == 'mouth' or name == 'left_eyebrow' or name == 'right_eyebrow':
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    # extract the ROI of the face region as a separate image
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    w += 30
                    h += 10
                    x -= 15
                    y -= 5

                    roi = image[y:y + h, x:x + w]
                    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

                    # show the particular face part
                    #cv2.imshow("ROI", roi)
                    #cv2.imshow("Image", clone)
                    #cv2.waitKey(0)
                ROIS.append(roi)


    return ROIS



def getROI2(image, shape):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    ROIS = []
    # load the input image, resize it, and convert it to grayscale


    image = imutils.resize(image, width=500)
    gray = image

    # loop over the face detections


    # loop over the face parts individually
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        if name == 'mouth' or name == 'left_eyebrow' or name == 'right_eyebrow':
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))


                w += 30
                h += 10
                x -= 15
                y -= 5

                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

                # show the particular face part
                #cv2.imshow("ROI", roi)
                #cv2.imshow("Image", clone)
                #cv2.waitKey(0)
            ROIS.append(roi)


    return ROIS

def getShape(image):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    image = imutils.resize(image, width=500)
    gray = image

    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    return shape