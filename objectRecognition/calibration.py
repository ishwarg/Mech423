import os
import cv2
from cv2 import aruco
import numpy as np

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

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist

def undistort_image(image, camera_matrix, dist_coeffs):

    return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)

def detect_aruco_markers(image_path, camera_matrix, dist_coeffs):
    # Load the image
    image = cv2.imread(image_path)

    # Undistort the image
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(0)
    


    # Convert the image to grayscale
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # Initialize the ArUco detector parameters
    parameters = aruco.DetectorParameters_create()

    # Detect ArUco markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers on the undistorted image
    image_markers = undistorted_image.copy()
    aruco.drawDetectedMarkers(image_markers, corners, ids)

    # Display the undistorted image with markers
    cv2.imshow("Undistorted Image with ArUco Markers", image_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the detected markers' information
    return corners, ids

if __name__ == "__main__":
    
    # Use relative paths based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_dir = os.path.join(script_dir, 'chess')
    image_path = os.path.join(script_dir, 'poolTable.jpg')

    # Create a list of file paths in the calibration directory
    calibration_images = [os.path.join(calibration_dir, file_name) for file_name in os.listdir(calibration_dir) if file_name.endswith(".jpg")]

    # Chessboard size (number of inner corners)
    chessboard_size = (7,7)
    
    # Calibrate the camera and obtain camera matrix and distortion coefficients
    camera_matrix, dist_coeffs = calibrate_camera(calibration_dir, chessboard_size)

    corners, ids = detect_aruco_markers(image_path, camera_matrix, dist_coeffs)

    if ids is not None and len(ids) == 4:
        print("Detected 4 ArUco markers:")
        for i in range(4):
            print(f"Marker ID {ids[i]} - Corners: {corners[i]}")
    else:
        print("Could not detect 4 ArUco markers in the image.")
    
