from imutils import face_utils
from functions import *
from svm import *
from preprocessing import getROI, getROI2, getShape
import numpy as np
import os
import imutils
import dlib
import cv2


#class to allign face based on landmarks

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape2 = shape
        shape = shape_to_np(shape)

        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

# function to calculate confusion matrix from pre extracted data descriptors and test data

fast_SVMpredict()

# main program

# cycling on test folder to process frame by frame videos

dest = 'test_folder'
folders = list(filter(lambda x: os.path.isdir(os.path.join(dest, x)), os.listdir(dest)))
for folder in folders:
    print("Processing folder : " + folder + " .........")

    # initializing 3 histogram relative to the main rois
    # first ROI is mouth/lips, second and third are left/right eyebrows

    hist_list0 = []
    hist_list1 = []
    hist_list2 = []
    counter = 0
    endingfolder(0)
    endingfolder(1)
    endingfolder(2)

    '''
    color = np.random.randint(0,1,(100,3))
    color1 = (66, 92, 238)
    color2 = (74, 74, 74)
    color3 = (69, 139, 0)
    '''

    folder = os.path.join('test_folder/' +folder)

    # load and store images from folder in test_folder

    images = load_images_from_folder(folder)

    # set sliding window size and timestep

    slidingWindow = 100
    window_counter = 0
    timestep = 5
    size = len(images)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    start_frame = 0
    end_frame = 100

    while((counter* timestep) <= size ):


        if window_counter*timestep == slidingWindow:
            window_counter = 0

            # overlap between sliding windows. first window is 0 - 100, second 60 - 160, third 120 - 220, ...
            counter -= 40/timestep
            #calcAndStoreHOF(firstRois, rois, hist_list)
            #calcAndStoreHOF(midROIS, rois, hist_list)
            #calcAndStoreHOF(firstRois, midROIS, hist_list)

            # write test data for the 3 rois and use SVM-classifier

            write_test(hist_list0, 0)
            write_test(hist_list1, 1)
            write_test(hist_list2, 2)
            SVM_predict(folder, start_frame, end_frame, 0)
            SVM_predict(folder, start_frame, end_frame, 1)
            SVM_predict(folder, start_frame, end_frame, 2)
            start_frame = end_frame - 40
            end_frame = start_frame + 100
            prepare_data(hist_list0, 0)
            prepare_data(hist_list1, 1)
            prepare_data(hist_list2, 2)
            endingWindow(0)
            endingWindow(1)
            endingWindow(2)
            hist_list0 = []
            hist_list1 = []
            hist_list2 = []

        #string = 'test{}.jpg'.format(counter)
        #path = os.path.join(string)
        #image = cv2.imread(path, 0)

        if counter == 0:
            currentImg = images[0]
        else:
            currentImg = images[counter*timestep - 1]

        if window_counter == 0:
            firstImg = currentImg.copy()

        image_copy = currentImg.copy()

        fa = FaceAligner(predictor, desiredFaceWidth=800)
        rects2 = detector(firstImg, 2)

        # loop over the face detections
        for rect in rects2:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(currentImg[y:y + h, x:x + w], width=250)
            faceAligned = fa.align(currentImg, currentImg, rect)
            #shape2 = predictor(image_copy, rect)

            # display the output images
            # cv2.imshow("Original", faceOrig)
            # cv2.imshow("Aligned", faceAligned)
            # cv2.waitKey(0)
        # show the output image with the face detections + facial landmarks

        rects = detector(faceAligned, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(faceAligned, rect)
            shape = face_utils.shape_to_np(shape)
            shape2 = predictor(faceAligned, rect)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            cv2.rectangle(faceAligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
            yy = y
            xx = x

            # show the face number
            cv2.putText(faceAligned, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            x = 0
            y = 0
            faceAlignedWithoutCircle = faceAligned.copy()
            for (x, y) in shape:
                cv2.circle(faceAligned, (x, y), 1, (0, 0, 255), -1)

        #rect_img3 = faceAligned[yy: yy + h, xx: xx + w]
        rect_img = faceAligned
        rect_img2 = faceAlignedWithoutCircle[yy: yy + h, xx: xx + w]
        #cv2.imshow('starter', image_copy)
        #cv2.waitKey(0)
        #cv2.imshow('faceDetected&landmarks', rect_img)
        #cv2.waitKey(0)
        #cv2.imshow('cropped&landmarks', rect_img3)
        #cv2.waitKey(0)

        if window_counter == 0:
            ss = getShape(rect_img2)

        rois = getROI2(rect_img2, ss)
        if window_counter == 0:
            firstRois = rois

        #cv2.imshow("Output", faceOrig)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imshow("ROIReceived", rois[0])
        #cv2.waitKey(0)

        if window_counter != 0:
            newImg = rect_img
            oold = oldImg[yy: yy + h, xx: xx + w]
            nnew = newImg[yy: yy + h, xx: xx + w]
            #old, new = getFlow(oldImg, newImg, prevPts)
            #imm = getDenseFlow(oold, nnew)

            calcAndStoreHOF(firstRois, rois, hist_list0, hist_list1, hist_list2)

            #cv2.imshow('dense', imm2)
            #prepare_data(hist_list)

            #mask = np.zeros_like(oldImg)
            # draw the tracks
            '''
            for i, (new, old) in enumerate(zip(new, old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color2, 2)
                frame = cv2.circle(oldImg, (a, b), 3, color1, -1)
                frame = cv2.circle(oldImg, (c, d), 3, color3, -1)
            img = cv2.add(frame, mask)
            img = img[yy: yy + h, xx: xx + w]
            #cv2.imshow('final', img)
            #cv2.waitKey(0)
            '''
            oldROIS = rois
            oldImg = rect_img
            #prevPts = shape_to_np2(shape2)
            #if window_counter * timestep == slidingWindow/2:
            #    midROIS = rois

        else:
            oldROIS = rois
            oldImg = rect_img
            #prevPts = shape_to_np2(shape2)
        counter += 1
        window_counter += 1

    difference = 19 - len(hist_list0)
    while difference > 0:
        calcAndStoreHOF(firstRois, rois, hist_list0, hist_list1, hist_list2)
        difference -= 1

    write_test(hist_list0, 0)
    write_test(hist_list1, 1)
    write_test(hist_list2, 2)
    SVM_predict(folder, start_frame, end_frame, 0)
    SVM_predict(folder, start_frame, end_frame, 1)
    SVM_predict(folder, start_frame, end_frame, 2)
    prepare_data(hist_list0, 0)
    prepare_data(hist_list1, 1)
    prepare_data(hist_list2, 2)
    endingWindow(0)
    endingWindow(1)
    endingWindow(2)