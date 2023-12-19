import os
import cv2
from cv2 import aruco
import numpy as np
from itertools import combinations
import math as m


def detectCue(image):
    # Load the image
    

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # Initialize the ArUco detector parameters
    parameters = aruco.DetectorParameters()

    # Instantiat an ArucoDetector object
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)
   

    if ids is not None:
        valid_ids = [4,5]
        valid_corners = [corner for corner, marker_id in zip(corners, ids) if marker_id[0] in valid_ids]
        valid_ids = [marker_id[0] for marker_id in ids if marker_id[0] in valid_ids]
        valid_ids = np.array(valid_ids)
        # Draw detected markers on the undistorted image
    
        aruco.drawDetectedMarkers(image, valid_corners, valid_ids)
        # Return the detected markers' information
        return valid_corners, valid_ids, image
    
    else:
        return [], [], image
    

    


    # Filter markers with IDs 0 to 3
    


def determineAngle(image, vector):


    corners, ids, image = detectCue(image)
    
    
    finalCorners = [(None)]*2
    if ids is not None and len(ids) == 2:
        #print("Detected 2 ArUco markers:")
        for i in range(2):
            
            #print(f"Marker ID {ids[i]} - Corners: {corners[i]}")
            
            if ids[i]==4:
                finalCorners[0]=tuple(corners[i][0][0])
            elif ids[i] == 5:
                finalCorners[1]=tuple(corners[i][0][0])
        vector = [finalCorners[0][0]-finalCorners[1][0], finalCorners[0][1]-finalCorners[1][1]]
        
    else:
        print("No Cue Detected")
        return vector
    vector = np.array(vector)
    magnitude = np.linalg.norm(vector)

    # Step 3: Create the unit vector by dividing each element by the magnitude
    unit_vector = vector / magnitude
    
    
    
    return unit_vector


