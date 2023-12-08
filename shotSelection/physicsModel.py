import numpy as np
import cv2
#from objectRecognition import cueAngle as ca , ballDetection as bd, calibration as cc
from objectRecognition.PoolTableConstants import *

#Function to see if there are ball collisions
#Input: 
# List of all ball coords
# list of balls to filter out (first ball is chosen as start position)
# Trajectory of ball
#Output
# True or false ball colision
def CheckCircleCollision(balls, ballIndices, traj):
    
    line_start = balls[ballIndices[0]]
    for i,circle_center in enumerate(balls):
        if i in ballIndices:
            continue

        # Vector from the start of the line to the circle center
        start_to_center = circle_center - line_start

        # Calculate the projection of start_to_center onto the line_vector
        projection = np.dot(start_to_center, traj) / np.dot(traj, traj)

        # Calculate the closest point on the line to the circle center
        closest_point = line_start + projection * traj

        # Calculate the distance between the closest point and the circle center
        distance = np.linalg.norm(closest_point - circle_center)
        
        # Check if the distance is less than or equal to the radius
        if distance <= RADIUS:
            return True
        
    return False

#Function to see if there are ball collisions
#Input: 
# List of all ball coords
# Ball to generate traj
# Trajectory of ball
#Output
# True or false ball colision   
def ObjectBallTraj(balls,ballIndex,pocketIndex):

    traj = POCKETS[pocketIndex] - balls[ballIndex]
    Collision = CheckCircleCollision(balls, [ballIndex,], traj)
    return Collision, traj

def CueBallTraj(balls,objectBallIndex,cueBallIndex,objectBallTraj):
    Line_end = objectBallTraj/(np.linalg.norm(objectBallTraj))*2*RADIUS + balls[objectBallIndex]
    Line_start = balls[cueBallIndex]
    traj = Line_start - Line_end.astype(int)
    Collision = CheckCircleCollision(balls, [cueBallIndex,objectBallIndex], traj)

    return Collision, traj

def DrawTraj(img,ball,traj):
    cv2.line(img,ball,ball+traj,(255,0,0),5)





