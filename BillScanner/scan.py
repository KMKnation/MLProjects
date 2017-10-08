# Import the modules
import cv2
import imutils
from transform import four_point_transform

from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import pickle

# Read the input image
image = cv2.imread("bill1.jpg")


# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

# cv2.imshow("Output", edged)
# cv2.waitKey(0)



# Threshold the image
ret, thresholdImage = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)


# Find contou   rs in the image
_, contours, hier = cv2.findContours(thresholdImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:1]


'''
pts = np.array(eval(args["coords"]), dtype="float32")

# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)
'''


# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(thresholdImage, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)


dst = cv2.cornerHarris(thresh,2,3,0.04)
#----result is dilated for marking the corners, not important-------------
dst = cv2.dilate(dst,None)
#----Threshold for an optimal value, it may vary depending on the image---
image[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()