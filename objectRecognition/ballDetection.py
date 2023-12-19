import cv2
import numpy as np
import matplotlib.pyplot as plt
#import calibration as cc
import os
from objectRecognition.PoolTableConstants import *

HUE_RANGES = [
    ("Red", (0, 15)),
    ("Yellow", (15, 45)),
    ("Green", (45, 75)),
    ("Blue", (75, 105)),
    ("Purple", (105, 135)),
    ("Red", (135, 180)),  # Handle wrap-around at the red boundary
]
'''
COLOUR_THRESHOLDS = {
    'YELLOW_UPPER':[0, 100, 100],
    'YELLOW_LOWER':[0, 100, 100],
    'BLUE_UPPER':[0, 100, 100],
    'BLUE_LOWER':[0, 100, 100],
    'RED_UPPER':[0, 100, 100],
    'RED_LOWER':[0, 100, 100],
    'PURPLE_UPPER':[0, 100, 100],
    'PURPLE_LOWER':[0, 100, 100],
    'ORANGE_UPPER':[0, 100, 100],
    'ORANGE_LOWER':[0, 100, 100],
    'GREEN_UPPER':[0, 100, 100],
    'GREEN_LOWER':[0, 100, 100],
    'MAROON_UPPER':[0, 100, 100],
    'MAROON_LOWER':[0, 100, 100],
    'BLACK_UPPER':[0, 100, 100],
    'BLACK_LOWER':[0, 100, 100],
    'CUE_UPPER':[0, 100, 100],
    'CUE_LOWER':[0, 100, 100],
}
'''

def FindBalls(ctrs, img):

    balls = [] # Dictionary for filtered contours
    for i,c in enumerate(ctrs): # for all contours
        
        M = cv2.moments(c)

        # If area smaller than a threshold filter it out
        if M["m00"]<np.pi*MINIMUM_RADIUS**2:
            continue

        # If area larger than a threshold filter it out
        if M["m00"]>np.pi*MAXIMUM_RADIUS**2:
            continue

        # If contour has straight lines filter it out
        lX=[x for [[x, _]] in c]
        lY=[y for [[_, y]] in c]
        
        if np.corrcoef(lX, lY)[0, 1]**2 > 0.75:
            [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            cv2.line(
                img,
                tuple(map(int, (x+vx*-1920, y+vy*-1920))),
                tuple(map(int, (x+vx*1920, y+vy*1920))),
                (255, 255, 255),
                15
            )
            continue

        # If contour is circular include it in final list
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        stdRadius = np.std([((x-ix)**2 + (y-iy)**2)**0.5 for ix, iy in zip(lX, lY)])

        if stdRadius < 15:
           # Sort the ball

           #Get img of specific ball
           ballImg = GenerateBallImg(c,img)
           #Debug
           cv2.imshow('res',ballImg)
           cv2.waitKey(0)

           #Check Colour against thresholds
           ballImg = cv2.cvtColor(ballImg,cv2.COLOR_BGR2HSV)   #convert color for color identification
           #color_label = classify_hue(ballImg_avgColor[0],HUE_RANGES)
           #balls[color_label] = [(x, y)]

           CheckStrips(ballImg)
           #Sort balls


           balls+=[np.array([x, y])]

    return balls

#Function to generate contours around all objects
#Input: Transformed image
#Output: contours
def GenerateContours(img,backgroundThreshold):
    # apply blur
    img_copy = np.copy(img)
    img_blur = cv2.GaussianBlur(img,(51,51),cv2.BORDER_DEFAULT) # blur applied

    # mask
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV) # convert to hsv
    if len(backgroundThreshold) == 4:
        mask1 = cv2.inRange(hsv, backgroundThreshold['lower'], backgroundThreshold['lowerMiddle']) # table's mask
        mask2 = cv2.inRange(hsv, backgroundThreshold['upperMiddle'], backgroundThreshold['upper']) # table's mask
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, backgroundThreshold['lower'], backgroundThreshold['upper']) # table's mask

    # filter mask
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # dilate->erode
    
    # apply threshold
    ret,mask_inv = cv2.threshold(mask,5,255,cv2.THRESH_BINARY_INV) # apply threshold

    # find contours and filter them
    ctrs, hierarchy = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours
    
    return ctrs

#Function to draw specific ball based on specific ball contour and pool table image
#Input: single contour and image of playfield
#Output: Cropped image of ball
def GenerateBallImg(c,img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(img_gray)
    
    cv2.drawContours(mask,[c],0,255,-1)

    ballImg = cv2.bitwise_and(img,img,mask = mask)

    return ballImg

#Function to check if ball is strips
#Input: Image of specfic ball
#Output: True or false
def CheckStrips(ballImg):
    ballImg = cv2.cvtColor(ballImg,cv2.COLOR_BGR2GRAY)
    cv2.imshow('window',ballImg)
    cv2.waitKey(0)
    ret,ballImg = cv2.threshold(ballImg,WHITE_THRESHOLD,255,cv2.THRESH_BINARY)

    cv2.imshow('window',ballImg)
    cv2.waitKey(0)

    if np.mean(ballImg) > int(PERCENT_WHITE_THRESHOLD*255):
        return True
    else:
        return False

def DrawBalls(balls,img):
    for i,(x,y) in enumerate(balls):
        cv2.putText(img,
            f'Ball id: {i}', (x+50, y-25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA)
        cv2.putText(img,
            f'({x}, {y})', (x+50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA)

def classify_hue(hue_value, hue_ranges):
    for label, (lower, upper) in hue_ranges:
        if lower <= hue_value <= upper:
            return label
    return "Unknown"

def GenerateBackgroundThresholds(img,num_samples):
    # Get the shape of the image
    rows, cols, channels = img.shape
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # Generate random pixel coordinates
    random_rows = np.random.randint(int(rows/20), int(rows*19/20), num_samples)
    random_cols = np.random.randint(int(cols/20), int(cols*19/20), num_samples)

    # Extract color values for the sampled pixels
    sampled_pixels = img_hsv[random_rows, random_cols]

    # Find the maximum and minimum color values
    backgroundThreshold = {}
    backgroundThreshold['upper'] = np.array([np.max(sampled_pixels[:,0])+10,200,255])
    if backgroundThreshold['upper'][0] > 180:
        backgroundThreshold['upper'][0] = backgroundThreshold['upper'][0] - 180
        backgroundThreshold['upperMiddle'] = np.array([0,100,150])
        backgroundThreshold['lowerMiddle'] =  np.array([180,200,255])
    backgroundThreshold['lower'] = np.array([np.min(sampled_pixels[:,0])-10,200,255])

    return backgroundThreshold


if __name__ == "__main__":

    #Load pool table & pool balls
    img = cv2.imread('Images\American-style_pool_table_diagram_(empty).png')
    #img = cv2.resize(img,IMGSIZE)
    img_copy = img
    cv2.namedWindow('res',cv2.WINDOW_KEEPRATIO)

    #Make sure both images succesfully loaded
    assert img is not None, "file could not be loaded"

    ctrs = GenerateContours(img)
    cv2.drawContours(img_copy,ctrs,-1,255,2)
    cv2.imshow('res',img_copy)
    cv2.waitKey(0)

    balls = FindBalls(ctrs, img)

    DrawBalls(balls,img)  
    cv2.imshow('res',img)
    cv2.waitKey(0)

