import os
import cv2
from cv2 import aruco
import numpy as np
from itertools import combinations
import math as m


def detectCue(image, camera_matrix, dist_coeffs):
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

    # Filter markers with IDs 0 to 3
    valid_ids = [4,5]
    valid_corners = [corner for corner, marker_id in zip(corners, ids) if marker_id[0] in valid_ids]
    valid_ids = [marker_id[0] for marker_id in ids if marker_id[0] in valid_ids]
    valid_ids = np.array(valid_ids)

    # Draw detected markers on the undistorted image
    image_markers = image.copy()
    aruco.drawDetectedMarkers(image_markers, valid_corners, valid_ids)

    # Return the detected markers' information
    return valid_corners, valid_ids, image_markers


def determineAngle(image, camera_matrix, dist_coeffs):


    corners, ids, newImage = detectCue(image, camera_matrix, dist_coeffs)

    
    finalCorners = [(None)]*2
    if ids is not None and len(ids) == 2:
        print("Detected 4 ArUco markers:")
        for i in range(2):
            
            print(f"Marker ID {ids[i]} - Corners: {corners[i]}")
            
            if ids[i]==4:
                finalCorners[0]=tuple(corners[i][0][0])
            elif ids[i] == 5:
                finalCorners[1]=tuple(corners[i][0][0])
        vector = (finalCorners[1][0]-finalCorners[0][0], finalCorners[1][1]-finalCorners[0][1])
        angle = m.atan(vector[1]/vector[0])
    else:
        print("No Cue Detected")
        angle = 0
    

    
    return angle


