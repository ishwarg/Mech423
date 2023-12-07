import numpy as np
import cv2
from objectRecognition import cueAngle as ca , ballDetection as bd, calibration as cc
from PoolTableConstants import *

RADIUS = cc.RADIUS# in mm
POCKETS = [(0,0),(0,MAX_HEIGHT),(MAX_WIDTH/2,MAX_HEIGHT),(MAX_WIDTH,MAX_HEIGHT),(MAX_WIDTH,0)]

#takes id of objectBall and cueBall
#takes list of ball coordinates and list of pocket coordinates
def idealAngle(objectBallIndex, cueBallIndex, balls, pocketIndex):
    Pockets = [(0,0)]

    objectCoordinates = balls[objectBallIndex]
    cueBallCoordinates = balls[cueBallIndex]

    ox = objectCoordinates[0]
    oy = objectCoordinates[1]
    cx = cueBallCoordinates[0]
    cy = cueBallCoordinates[1]

    return None

def ObjectBallTraj(balls,ballIndex,pocket):
    traj = Pockets[pocketIndex] - balls[ballIndex]
    return traj






