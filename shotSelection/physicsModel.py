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
def CheckCircleCollision(balls, startPoint, traj):

    for i,circle_center in enumerate(balls):
        # Ignore these indices
        if np.array_equal(circle_center,startPoint):
            continue

        # Vector from the start of the line to the circle center
        start_to_center = circle_center - startPoint

        # Calculate the projection of start_to_center onto the line_vector
        projection = np.dot(start_to_center, traj) / np.dot(traj, traj)

        # Check if the projection is in the bounds of the line segment
        if 0 <= projection <= 1:
            # Calculate the closest point on the line to the circle center
            closest_point = startPoint + projection * traj

            # Calculate the distance between the closest point and the circle center
            distance = np.linalg.norm(closest_point - circle_center)

            # Check if the distance is less than or equal to the circle radius
            if distance <= RADIUS*2:
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

    traj_list = [(POCKETS[pocketIndex] - balls[ballIndex])]
    Collision = CheckCircleCollision(balls, balls[ballIndex], traj_list[0])
    if Collision:
        traj_list = ProcessBounces(balls[ballIndex],POCKETS[pocketIndex],balls)
    return traj_list

def CueBallTraj(balls,objectBallIndex,cueBallIndex,objectBallTraj):
    trajDir =  objectBallTraj/ np.linalg.norm(objectBallTraj)
    Line_end = -trajDir*2*(RADIUS + 1) + balls[objectBallIndex]
    Line_start = balls[cueBallIndex]
    traj_list = [(Line_end.astype(int) - Line_start)]
    Collision = CheckCircleCollision(balls, balls[cueBallIndex], traj_list[0])
    if Collision:
        traj_list = ProcessBounces(Line_start,Line_end.astype(int),balls)

    return traj_list

def ProcessBounces(startPoint,endPoint,balls):
    trajectories = []
    if endPoint[1] == 0:
        collision, bounceTraj = BottomBounce(startPoint,endPoint,balls)
        if collision == False:
            trajectories.append(bounceTraj)
        
        # Do Left and right Bounces
        if endPoint[0] == 0:
            collision, bounceTraj = RightBounce(startPoint,endPoint,balls)
            if collision == False:
                trajectories.append(bounceTraj)
        
        elif endPoint[0] == MAX_WIDTH:
            collision, bounceTraj = LeftBounce(startPoint,endPoint,balls)
            if collision == False:
                trajectories.append(bounceTraj)
        
        else:
            collision, bounceTraj = RightBounce(startPoint,endPoint,balls)
            if collision == False:
                trajectories.append(bounceTraj)
            collision, bounceTraj = LeftBounce(startPoint,endPoint,balls)
            if collision == False:
                trajectories.append(bounceTraj)
    
    elif endPoint[1] == MAX_HEIGHT:
        collision, bounceTraj = TopBounce(startPoint,endPoint,balls)
        if collision == False:
            trajectories.append(bounceTraj)

        # Do Left and right Bounces
        if endPoint[0] == 0:
            collision, bounceTraj = RightBounce(startPoint,endPoint,balls)
            if collision == False:
                trajectories.append(bounceTraj)
        
        elif endPoint[0] == MAX_WIDTH:
            collision, bounceTraj = LeftBounce(startPoint,endPoint,balls)
            if collision == False:
                trajectories.append(bounceTraj)
        
        else:
            collision, bounceTraj = RightBounce(startPoint,endPoint,balls)
            if collision == False:
                trajectories.append(bounceTraj)
            collision, bounceTraj = LeftBounce(startPoint,endPoint,balls)
            if collision == False:
                trajectories.append(bounceTraj)
    else:
        collision, bounceTraj = BottomBounce(startPoint,endPoint,balls)
        if collision == False:
           trajectories.append(bounceTraj)
        
        collision, bounceTraj = TopBounce(startPoint,endPoint,balls)
        if collision == False:
            trajectories.append(bounceTraj)
        
        collision, bounceTraj = LeftBounce(startPoint,endPoint,balls)
        if collision == False:
            trajectories.append(bounceTraj)
        
        collision, bounceTraj = RightBounce(startPoint,endPoint,balls)
        if collision == False:
            trajectories.append(bounceTraj)

    min_magnitude = float('inf')
    for i, traj in enumerate(trajectories, start=1):
    # Calculate the Euclidean distance between consecutive points in the trajectory
        traj = np.array(traj)
        distances = np.linalg.norm(traj, axis=1)
    
        # Sum the distances to get the magnitude
        magnitude = np.sum(distances)
    
        # Check if the current trajectory has a lower magnitude
        if magnitude < min_magnitude:
            min_magnitude = magnitude
            min_magnitude_trajectory = traj
    return  min_magnitude_trajectory
        
        


def BottomBounce(startPoint,endPoint,balls):
    P = np.array([0,MAX_HEIGHT])
    P[0] = ((endPoint[1]-P[1])*startPoint[0] + (startPoint[1]-P[1])*endPoint[0])/(endPoint[1]+startPoint[1] - 2*P[1])
    traj = [(P - startPoint),(endPoint - P)]
    
    collision = False
    collision = CheckCircleCollision(balls, startPoint, traj[0])
    collision = CheckCircleCollision(balls, P, traj[1])
    return collision, traj

def TopBounce(startPoint,endPoint,balls):
    P = np.array([0,0])
    P[0] = ((endPoint[1]-P[1])*startPoint[0] + (startPoint[1]-P[1])*endPoint[0])/(endPoint[1]+startPoint[1] - 2*P[1])
    traj = [(P - startPoint),(endPoint - P)]
    
    collision = False
    collision = CheckCircleCollision(balls, startPoint, traj[0])
    collision = CheckCircleCollision(balls, P, traj[1])
    return collision, traj

def RightBounce(startPoint,endPoint,balls):
    P = np.array([MAX_WIDTH,0])
    P[1] = ((endPoint[0]-P[0])*startPoint[1] + (startPoint[0]-P[0])*endPoint[1])/(endPoint[0]+startPoint[0] - 2*P[0])
    traj = [(P - startPoint),(endPoint - P)]
    
    collision = False
    collision = CheckCircleCollision(balls, startPoint, traj[0])
    collision = CheckCircleCollision(balls, P, traj[1])
    return collision, traj

def LeftBounce(startPoint,endPoint,balls):
    P = np.array([0,0])
    P[1] = ((endPoint[0]-P[0])*startPoint[1] + (startPoint[0]-P[0])*endPoint[1])/(endPoint[0]+startPoint[0] - 2*P[0])
    traj = [(P - startPoint),(endPoint - P)]
    
    collision = False
    collision = CheckCircleCollision(balls, startPoint, traj[0])
    collision = CheckCircleCollision(balls, P, traj[1])
    return collision, traj

def DrawTraj(img, startPoint, traj_list):
    current_point = startPoint  # Initialize current_point with startPoint

    for traj in traj_list:
        end_point = current_point + np.array(traj)  # Calculate end_point for the current vector
        
        # Draw a line from current_point to end_point
        cv2.line(img, tuple(current_point), tuple(end_point), (255, 255, 255), 2)
        
        # Draw a circle at the end_point
        cv2.circle(img, tuple(end_point), RADIUS, (255, 255, 255), 1)

        # Update current_point for the next iteration
        current_point = end_point





