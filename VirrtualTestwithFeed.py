#import objectRecognition.calibration as cc
#import objectRecognition.cueAngle as ca
import objectRecognition.ballDetection as bd
from shotSelection.physicsModel import *
from objectRecognition.PoolTableConstants import *
import os
import cv2
import numpy as np
import objectRecognition.calibration as cc
import traceback

BACKGROUND_IMG = 'image.png'
INPLAY_IMG = 'IMG_0113.jpg'
#Ball and cue selection
OBJECT_BALL_INDEX = 1
POCKET_INDEX = 2
CUE_BALL_INDEX = 3
BACKGROUND_THRESHOLDS = {
    'upper': np.array([8,200,255]),
    'upperMiddle':np.array([0,100,150]),
    'lowerMiddle': np.array([180,200,255]),
    'lower':np.array([172,100,150])
}

#camera_matrix, dist_coeffs, finalCorners, topDown = cc.initialCalibration()
# angle = ca.determineAngle(topDown, camera_matrix, dist_coeffs)
# print(angle)

#Load Background Image
# script_dir = os.path.dirname(os.path.abspath(__file__))
# image_path = os.path.join(script_dir, BACKGROUND_IMG)
# img =cv2.imread(image_path)
# assert img is not None, "file could not be loaded"
# cv2.namedWindow('window',cv2.WINDOW_KEEPRATIO)
# #Debug
# cv2.imshow('window',img)
# cv2.waitKey(0)

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


file_path = "/Users/ishwarjotgrewal/Desktop/Mech423/objectRecognition/calibration_data.npz"

data = np.load(file_path)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()



# Set the window name
window_name = "Webcam Feed"

# Create a window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Break the loop if reading the frame fails
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame
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
            warped = cc.generate_top_down_view(image_markers, finalCorners, MAX_WIDTH, MAX_HEIGHT)
            
            ctrs = bd.GenerateContours(warped, BACKGROUND_THRESHOLDS)
            cv2.drawContours(warped,ctrs,-1,255,2)
            balls = bd.FindBalls(ctrs, warped)
            bd.DrawBalls(balls,warped)
            cv2.imshow(window_name, warped)

                
        else:
            print("Could not detect 4 ArUco markers in the image.")
            cv2.imshow(window_name, frame)
        
        
    except Exception as e:
            print(f"Error in detect_aruco_markers: {e}")
            
    
    

    # Save the frame as a JPEG image when 'w' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite('captured_frame.jpg', warped)
        print("Frame saved as 'captured_frame.jpg'")

    # Exit the loop when 'q' key is pressed
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the window
cap.release()

cv2.destroyAllWindows()

# #Generate contours
# ctrs = bd.GenerateContours(img, BACKGROUND_THRESHOLDS)
# #Debug
# img_copy = np.copy(img)
# cv2.drawContours(img_copy,ctrs,-1,255,2)
# cv2.imshow('window',img_copy)
# cv2.waitKey(0)

# #Find ball Indices
# balls = bd.FindBalls(ctrs, img)
# img_ids = np.copy(img)
# bd.DrawBalls(balls,img_ids) 
# #Debug 
# #cv2.imshow('window',img_ids)
# #cv2.waitKey(0)

# #Find ball trajectories
# #user_input = input("Enter something: ")

# # Print the user input
# while(True):
    
#     cv2.imshow('window',img_ids)
#     cv2.waitKey(0)

#     POCKET_INDEX = int(input("Select pocket index:"))
#     print("You entered:", POCKET_INDEX)

#     OBJECT_BALL_INDEX = int(input("Select object ball index:"))
#     print("You entered:", OBJECT_BALL_INDEX)

#     CUE_BALL_INDEX = int(input("Select cue ball index:"))
#     print("You entered:", CUE_BALL_INDEX)

#     collisionObject,objectBallTraj = pm.ObjectBallTraj(balls,OBJECT_BALL_INDEX,POCKET_INDEX)
#     collisionCue,cueBallTraj = pm.CueBallTraj(balls,OBJECT_BALL_INDEX,CUE_BALL_INDEX,objectBallTraj)
#     #debug
#     if collisionObject:
#         print("Object ball has a collision")
#     if collisionCue:
#         print("Cue ball has a collision")
#     img_copy = np.copy(img)
#     pm.DrawTraj(img_copy,balls[OBJECT_BALL_INDEX],objectBallTraj) 
#     pm.DrawTraj(img_copy,balls[CUE_BALL_INDEX],cueBallTraj)  
#     cv2.imshow('window',img_copy)
#     cv2.waitKey(0)
#     user_input = input("Enter something ('exit' to break): ")

#     if user_input.lower() == 'exit':
#         break


