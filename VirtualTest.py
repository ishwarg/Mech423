#import objectRecognition.calibration as cc
#import objectRecognition.cueAngle as ca
import objectRecognition.ballDetection as bd
import shotSelection.physicsModel as pm
from objectRecognition.PoolTableConstants import *
import os
import cv2

BACKGROUND_IMG = 'Somethingelse.jpg'
INPLAY_IMG = 'IMG_0113.jpg'
#Ball and cue selection
OBJECT_BALL_INDEX = 1
POCKET_INDEX = 2
CUE_BALL_INDEX = 3

#camera_matrix, dist_coeffs, finalCorners, topDown = cc.initialCalibration()
# angle = ca.determineAngle(topDown, camera_matrix, dist_coeffs)
# print(angle)

#Load Background Image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, BACKGROUND_IMG)
img =cv2.imread(image_path)
assert img is not None, "file could not be loaded"
cv2.namedWindow('window',cv2.WINDOW_KEEPRATIO)
#Debug
cv2.imshow('window',img)
cv2.waitKey(0)

'''
#Flatten & crop image
corners, warped = cc.tableDetection(image, camera_matrix, dist_coeffs)
#Debug
cv2.imshow('window',image)
cv2.waitKey(0)

#Get background thresholds
backgroundThreshold = bd.GenerateBackgroundThresholds(image, 10)
#Debug
print(backgroundThreshold)

#Load Game Image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, INPLAY_IMG)
image =cv2.imread(image_path)
assert image is not None, "file could not be loaded"
cv2.namedWindow('window',cv2.WINDOW_KEEPRATIO)
#Debug
cv2.imshow('window',image)
cv2.waitKey(0)
'''
#Generate contours
ctrs = bd.GenerateContours(img,BACKGROUND_THRESHOLDS)
#Debug
img_copy = np.copy(img)
cv2.drawContours(img_copy,ctrs,-1,255,2)
cv2.imshow('window',img_copy)
cv2.waitKey(0)

#Find ball Indices
balls = bd.FindBalls(ctrs, img)
#Debug
img_copy = np.copy(img)
bd.DrawBalls(balls,img_copy)  
cv2.imshow('window',img_copy)
cv2.waitKey(0)

#Find ball trajectories
collisionObject,objectBallTraj = pm.ObjectBallTraj(balls,OBJECT_BALL_INDEX,POCKET_INDEX)
collisionCue,cueBallTraj = pm.CueBallTraj(balls,OBJECT_BALL_INDEX,CUE_BALL_INDEX,objectBallTraj)
#debug
if collisionObject:
    print("Object ball has a collision")
if collisionCue:
    print("Cue ball has a collision")
img_copy = np.copy(img)
pm.DrawTraj(img_copy,balls[OBJECT_BALL_INDEX],objectBallTraj) 
pm.DrawTraj(img_copy,balls[CUE_BALL_INDEX],cueBallTraj)  
cv2.imshow('window',img_copy)
cv2.waitKey(0)


