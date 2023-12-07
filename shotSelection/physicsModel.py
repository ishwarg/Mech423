import numpy as np
import cv2
from objectRecognition import cueAngle as ca , ballDetection as bd, calibration as cc

RADIUS = cc.RADIUS# in mm

#takes id of objectBall and cueBall
#takes list of ball coordinates and list of pocket coordinates
def idealAngle(objectBallIndex, cueBallIndex, ballCoordinates, pockets, pocketIndex):

    objectCoordinates = ballCoordinates[objectBallIndex]
    cueBallCoordinates = ballCoordinates[cueBallIndex]
    targetPocket = pockets[pocketIndex]

    ox = objectCoordinates[0]
    oy = objectCoordinates[1]
    cx = cueBallCoordinates[0]
    cy = cueBallCoordinates[1]
    px = targetPocket[0]
    py = targetPocket[1]




    return None






