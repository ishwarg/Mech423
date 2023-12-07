import cv2
import numpy as np
import matplotlib.pyplot as plt
import calibration as cc
import os

# Constants
WHITE_THRESHOLD = 10    #Threshold for the whit part of the ball
PERCENT_WHITE_THRESHOLD = 0.5   #Threshold for what percentage of the ball is white before it is considered a stiped ball
MINIMUM_RADIUS = 25     #Minimum pool ball radius to expect in the image
MAXIMUM_RADIUS = 80     #Max pool ball radius to expect in the image
IMGSIZE = [1120, 2240]
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
BACKGROUND_THRESHOLD = {
    'UPPER': np.array([70, 255,240]),
    'LOWER': np.array([45, 100,135])
}

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

        if stdRadius < 3:
           # Sort the ball

           #Get img of specific ball
           ballImg = GenerateBallImg(c,img)
           cv2.imshow('res',ballImg)
           cv2.waitKey(0)

           #Check Colour against thresholds
           ballImg = cv2.cvtColor(ballImg,cv2.COLOR_BGR2HSV)   #convert color for color identification
           ballImg_avgColor = cv2.mean(ballImg)
           #if CheckStrips(ballImg):
           #Sort balls


           balls+=[(x, y)]

    return balls

#Function to generate contours around all objects
#Input: Transformed image
#Output: contours
def GenerateContours(img):
    # apply blur
    img_blur = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT) # blur applied

    # mask
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV) # convert to hsv
    mask = cv2.inRange(hsv, BACKGROUND_THRESHOLD['LOWER'], BACKGROUND_THRESHOLD['UPPER']) # table's mask

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
    ret,ballImg = cv2.threshold(ballImg,WHITE_THRESHOLD,255,cv2.THRESH_BINARY)

    if np.mean(ballImg) > int(PERCENT_WHITE_THRESHOLD*255):
        return True
    else:
        return False

if __name__ == "__main__":

    #Load pool table & pool balls
    img = cv2.imread('PoolTableWithBallsWarped.jpg')
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

    for i,(x,y) in enumerate(balls):
        cv2.putText(img,
            f'Ball id: {i}', (x+50, y-25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA)
        cv2.putText(img,
            f'({x}, {y})', (x+50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA)
        
    cv2.imshow('res',img)
    cv2.waitKey(0)
