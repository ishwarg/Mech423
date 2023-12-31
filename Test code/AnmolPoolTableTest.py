import numpy as np
import cv2 as cv
import sys
import os
from matplotlib import pyplot as plt

"""
This code places pictures of pool balls on top of a pool table
"""

directoryPath = r'C:\Users\anmol\Documents\UBC\MECH 4\MECH 423\Labs\Final Project\Mech423\Images';

#Load pool table & pool balls
imgPoolTable = cv.imread(os.path.join(directoryPath,'American-style_pool_table_diagram_(empty).png'))
imgPoolBalls = cv.imread(os.path.join(directoryPath,'Poolballs.png'))

#Make sure both images succesfully loaded
assert imgPoolTable is not None, "file could not be loaded"
assert imgPoolBalls is not None, "file could not be loaded"

#define region of interest (ROI) at top left corner of pool table size of pool balls img
rows,cols,channels = imgPoolBalls.shape
imgPoolBalls = cv.resize(imgPoolBalls, (200,200))
rows,cols,channels = imgPoolBalls.shape
roi = imgPoolTable[0:rows,0:cols]

#Create mask of image
imgPoolBallsBlue = imgPoolBalls[:,:,0]
imgPoolBallsGreen = imgPoolBalls[:,:,1]
imgPoolBallsRed = imgPoolBalls[:,:,2]
imgPoolBallsHSV = cv.cvtColor(imgPoolBalls,cv.COLOR_BGR2HSV)
# Define the range of green in HSV
lower_green = np.array([57, 190, 136])
upper_green = np.array([63, 196, 142])
mask_inv = cv.inRange(imgPoolBallsHSV, lower_green, upper_green)
mask = cv.bitwise_not(mask_inv)

# Now black-out the area of logo in ROI
imgPoolTable_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
imgPoolBalls_fg = cv.bitwise_and(imgPoolBalls,imgPoolBalls,mask = mask)

#Display an image
dst = cv.add(imgPoolTable_bg,imgPoolBalls_fg)
imgPoolTable[0:rows, 0:cols ] = dst
cv.imshow('res',imgPoolTable)
cv.waitKey(0)
cv.destroyAllWindows()