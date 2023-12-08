import cv2
import calibration as cc
import os
import numpy as np
from PoolTableConstants import *
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'calibration_data.npz')

data = np.load(file_path)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']
# Create a VideoCapture object
cap = cv2.VideoCapture(1)  # 0 corresponds to the default camera (usually the built-in webcam)

# Check if the camera opened successfully
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
        
    else:
    # Display the frame
        try:
            corners, ids, image_markers=cc.detect_aruco_markers(frame, camera_matrix, dist_coeffs)
            finalCorners = [(None)]*4
            print(ids)

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
                cv2.imshow(window_name, frame)
            warped = cc.generate_top_down_view(image_markers, finalCorners, MAX_WIDTH, MAX_HEIGHT)
            cv2.imshow(window_name, warped)
        except Exception as e:
                traceback_details = traceback.format_exc()
                print(f"Error in detect_aruco_markers: {e}\n{traceback_details}")
            
    
    

    # Save the frame as a JPEG image when 'w' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite('captured_frame.jpg', frame)
        print("Frame saved as 'captured_frame.jpg'")

    # Exit the loop when 'q' key is pressed
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the window
cap.release()
cv2.destroyAllWindows()
