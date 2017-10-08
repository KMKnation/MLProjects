# Import the modules
import cv2
import imutils
from transform import four_point_transform

from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import pickle

# Read the input image
image = cv2.imread("example.jpg")


# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)  #new added
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

# cv2.imshow("Output", edged)
# cv2.waitKey(0)



# Threshold the image
ret, thresholdImage = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)


# Find contours in the image
_, contours, hier = cv2.findContours(thresholdImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:1]



# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(thresholdImage, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('thresh',thresh)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# croping image by getting throsold border
_a, a_contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
a_contours= sorted(a_contours, key = cv2.contourArea, reverse = True)[:1]  #will sort all inner edges
cnt = a_contours[0]
x,y,w,h = cv2.boundingRect(cnt)
crop = image[y:y+h,x:x+w]

cv2.imshow('cropped',crop)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()




# Find contours in the image
t_, t_contours, t_hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
t_contours= sorted(t_contours, key = cv2.contourArea, reverse = True)[:1]

displayCnt = None

# loop over the contours
for c in t_contours:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the contour has four vertices, then we have found
    # the thermostat display
    print(len(approx))
    if len(approx) == 4:
        displayCnt = approx
        break




# extract the thermostat display, apply a perspective transform
# to it
# warped = four_point_transform(gray, displayCnt.reshape(4, 2))
# output = four_point_transform(image, displayCnt.reshape(4, 2))


# cv2.imshow('dst',warped)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

#corner harris to find desired corner of image
dst = cv2.cornerHarris(thresh,2,3,0.04)
#----result is dilated for marking the corners, not important-------------
dst = cv2.dilate(dst,None)
#----Threshold for an optimal value, it may vary depending on the image---
image[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst corner haris',image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

