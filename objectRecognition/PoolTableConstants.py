# Constants
import numpy as np
WHITE_THRESHOLD = 10    #Threshold for the whit part of the ball
PERCENT_WHITE_THRESHOLD = 0.5   #Threshold for what percentage of the ball is white before it is considered a stiped ball
MINIMUM_RADIUS = 25     #Minimum pool ball radius to expect in the image
MAXIMUM_RADIUS = 80     #Max pool ball radius to expect in the image
MAX_WIDTH = 2000
MAX_HEIGHT = 1000
IMGSIZE = [MAX_HEIGHT, MAX_WIDTH]
TABLE_WIDTH = 224 #cm
TABLE_HEIGHT = 112  #cm
RADIUS = int(MAX_HEIGHT*1/20)
XOFFSET_LENGTH = 6 #cm --> THIS NEEDS TO BE UPDATED STILL!
YOFFSET_LENGTH = 5.6
POCKETS = [
    np.array([0,0]),
    np.array([0,MAX_HEIGHT]),
    np.array([int(MAX_WIDTH/2),MAX_HEIGHT]),
    np.array([MAX_WIDTH,MAX_HEIGHT]),
    np.array([MAX_WIDTH,0]),
    np.array([int(MAX_WIDTH/2),0])]
BACKGROUND_THRESHOLDS ={
    "upper":np.array([76, 150, 200]),
    "lower":np.array([60, 90, 100])
}
XOFFSET = int(MAX_WIDTH/TABLE_WIDTH*XOFFSET_LENGTH)
YOFFSET = int(MAX_HEIGHT/TABLE_HEIGHT*YOFFSET_LENGTH)