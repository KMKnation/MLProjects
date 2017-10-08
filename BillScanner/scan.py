# Import the modules
import cv2
import imutils
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


cv2.imshow("Output", thresholdImage)
cv2.waitKey(0)

# Find contours in the image
# _, ctrs, hier = cv2.findContours(thresholdImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)