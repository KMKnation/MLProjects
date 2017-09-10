# import the necessary packages
import imutils
from transform import four_point_transform
import numpy as np
import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords",
                help="comma seperated list of source points")
args = vars(ap.parse_args())

# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)


#
# # convert the warped image to grayscale and then adjust
# # the intensity of the pixels to have minimum and maximum
# # values of 0 and 255, respectively
# warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# warp = exposure.rescale_intensity(warp, out_range = (0, 255))
#


# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)