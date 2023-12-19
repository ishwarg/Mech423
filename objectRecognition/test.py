import calibration as cc
import cueAngle as ca
import ballDetection as bd

import os
import cv2
import numpy as np
import traceback

BACKGROUND_THRESHOLDS = {
    'upper': np.array([8,200,255]),
    'upperMiddle':np.array([0,100,150]),
    'lowerMiddle': np.array([180,200,255]),
    'lower':np.array([172,100,150])
}

#camera_matrix, dist_coeffs, finalCorners, warped = cc.initialCalibration()

# print(camera_matrix, dist_coeffs)

# camera_matrix, dist_coeffs = cc.video_calibration((7,7))
# np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'calibration_data.npz')

data = np.load(file_path)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']


# script_dir = os.path.dirname(os.path.abspath(__file__))
# image_path = os.path.join(script_dir, 'captured_frame.jpg')
# image =cv2.imread(image_path)

# finalCorners, warped=cc.tableDetection(image, camera_matrix, dist_coeffs)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")

    try:
        
        corners, ids, image_markers=cc.detect_aruco_markers(frame, camera_matrix, dist_coeffs)

        finalCorners = [(None)]*4

        if ids is not None and len(ids) == 4:
            #print("Detected 4 ArUco markers:")
            for i in range(4):
                
                #print(f"Marker ID {ids[i]} - Corners: {corners[i]}")
                
                if ids[i]==0:
                    finalCorners[int(ids[i])]=tuple(corners[i][0][0])
                elif ids[i] == 1:
                    finalCorners[int(ids[i])]=tuple(corners[i][0][0])
                elif ids[i] == 2:
                    finalCorners[int(ids[i])]=tuple(corners[i][0][0])
                elif ids[i] == 3:
                    
                    finalCorners[int(ids[i])]=tuple(corners[i][0][0])
                
        else:
            print("Could not detect 4 ArUco markers in the image.")
        warped = cc.generate_top_down_view(image_markers, finalCorners, 2000, 1000)
        ctrs = bd.GenerateContours(warped, BACKGROUND_THRESHOLDS)
        cv2.drawContours(warped,ctrs,-1,255,2)
        balls = bd.FindBalls(ctrs, warped)
        bd.DrawBalls(balls,warped)
        cv2.imshow("window", warped)

    except Exception as e:
                        traceback_details = traceback.format_exc()
                        print(f"Error in detect_aruco_markers: {e}\n{traceback_details}")






