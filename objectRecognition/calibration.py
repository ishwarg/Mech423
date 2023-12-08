import os
import cv2
from cv2 import aruco
import numpy as np
from itertools import combinations
from PoolTableConstants import *
import time

def getPositionMM(x,y):
    actualx = x*float(TABLE_LENGTH)/float(MAX_WIDTH)
    actualy = y*float(TABLE_WIDTH)/float(MAX_HEIGHT)
    return actualx, actualy

def calibrate_camera(calibration_dir, chessboard_size):
    obj_points = []
    img_points = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    for file_name in os.listdir(calibration_dir):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(calibration_dir, file_name)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                obj_points.append(objp)
                img_points.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None, flags=cv2.CALIB_RATIONAL_MODEL)

    return mtx, dist


def video_calibration(chessboard_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Create a VideoCapture object
    cap = cv2.VideoCapture(1)  # 0 corresponds to the default camera (usually the built-in webcam)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Break the loop if reading the frame fails
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If corners are found, refine them using cornerSubPix and draw them on the image
        if ret:
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
            
        # Display the frame
        cv2.imshow("Calibration", frame)
      

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist
    

def detect_aruco_markers(image, camera_matrix, dist_coeffs):
    
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    

    # Convert the image to grayscale
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)


    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # Initialize the ArUco detector parameters
    parameters = aruco.DetectorParameters()

    # Instantiat an ArucoDetector object
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)
   
    # Filter markers with IDs 0 to 3
    valid_ids = [0, 1, 2, 3]
    valid_corners = [corner for corner, marker_id in zip(corners, ids) if marker_id[0] in valid_ids]
    valid_ids = [marker_id[0] for marker_id in ids if marker_id[0] in valid_ids]
    valid_ids = np.array(valid_ids)

    # Draw detected markers on the undistorted image
    image_markers = undistorted_image.copy()
    aruco.drawDetectedMarkers(image_markers, valid_corners, valid_ids)

    cv2.imshow("Aruco Detection", image_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Return the detected markers' information
    return valid_corners, valid_ids, image_markers

    
    


def generate_top_down_view(image, extremeCorners, maxWidth, maxHeight):
  
    # Extract the ordered corners
    tl, bl, br, tr = extremeCorners
 
    dst = np.array([
		[OFFSET, OFFSET],
		[maxWidth - 1-OFFSET, OFFSET],
		[maxWidth - 1-OFFSET, maxHeight - 1-OFFSET],
		[OFFSET, maxHeight - 1-OFFSET]], dtype = "float32")
    # compute the perspective transform matrix and then apply it


    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags = cv2.INTER_LINEAR)



#    # Visualization (Optional)
#     cv2.imshow('Original Image', image)
#     cv2.imshow('Top-Down View', warped)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    return warped


def initialCalibration():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_dir = os.path.join(script_dir, 'chess')
    image_path = os.path.join(script_dir, 'IMG_0113.jpg')
    image =cv2.imread(image_path)

    # Create a list of file paths in the calibration directory
   
    # Chessboard size (number of inner corners)
    chessboard_size = (7,7)

    
    # Calibrate the camera and obtain camera matrix and distortion coefficients
    camera_matrix, dist_coeffs = calibrate_camera(calibration_dir, chessboard_size)

    corners, ids, markersDetectedImage = detect_aruco_markers(image, camera_matrix, dist_coeffs)
    

    finalCorners = [(None)]*4

    if ids is not None and len(ids) == 4:
        print("Detected 4 ArUco markers:")
        for i in range(4):
            
            #print(f"Marker ID {ids[i]} - Corners: {corners[i]}")
            
            if ids[i]==0:
                finalCorners[int(ids[i])]=tuple(corners[i][0][0])
            elif ids[i] == 1:
                finalCorners[int(ids[i])]=tuple(corners[i][0][3])
            elif ids[i] == 2:
                finalCorners[int(ids[i])]=tuple(corners[i][0][2])
            elif ids[i] == 3:
                
                finalCorners[int(ids[i])]=tuple(corners[i][0][1])
            
    else:
        print("Could not detect 4 ArUco markers in the image.")
    
    
    # print(finalCorners)
    
    warped = generate_top_down_view(markersDetectedImage, finalCorners, MAX_WIDTH, MAX_HEIGHT)

    return camera_matrix, dist_coeffs, finalCorners, warped

def tableDetection(image, camera_matrix, dist_coeffs):
    corners, ids, markersDetectedImage = detect_aruco_markers(image, camera_matrix, dist_coeffs)
    

    finalCorners = [(None)]*4

    if ids is not None and len(ids) == 4:
        print("Detected 4 ArUco markers:")
        for i in range(4):
            
            print(f"Marker ID {ids[i]} - Corners: {corners[i]}")
            
            if ids[i]==0:
                finalCorners[int(ids[i])]=tuple(corners[i][0][0])
            elif ids[i] == 1:
                finalCorners[int(ids[i])]=tuple(corners[i][0][3])
            elif ids[i] == 2:
                finalCorners[int(ids[i])]=tuple(corners[i][0][2])
            elif ids[i] == 3:
                
                finalCorners[int(ids[i])]=tuple(corners[i][0][1])
            
    else:
        print("Could not detect 4 ArUco markers in the image.")
    
    warped = generate_top_down_view(markersDetectedImage, finalCorners, MAX_WIDTH, MAX_HEIGHT)
    
    cv2.imshow('Top-Down View', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return finalCorners, warped



if __name__ == "__main__":
    initialCalibration()