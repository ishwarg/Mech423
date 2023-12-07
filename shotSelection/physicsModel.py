import numpy as np
import cv2
from objectRecognition import cueAngle as ca , ballDetection as bd, calibration as cc

RADIUS = cc.RADIUS# in mm

#takes id of objectBall and cueBall
#takes list of ball coordinates and list of pocket coordinates
def idealAngle(objectBall, cueBall, ballCoordinates, pockets):

    objectCoordinates = ballCoordinates[objectBall]
    cueBallCoordinates = ballCoordinates[cueBall]

    ox = objectCoordinates[0]
    oy = objectCoordinates[1]
    cx = cueBallCoordinates[0]
    cy = cueBallCoordinates[1]
    
    ox, oy = cc.getPositionMM(ox, oy)
    cx, cy = cc.getPositionMM(cx,cy)

    pocketPositionsMM = [None]*6

    for pocket in pockets:
        x,y = cc.getPositionMM(pocket[0], pocket[1])
        pocketPositionsMM.append((x,y))
    

    



    return None






