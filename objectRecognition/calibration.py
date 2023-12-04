import os
import cv2
from cv2 import aruco
import numpy as np
from itertools import combinations


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

def detect_aruco_markers(image_path, camera_matrix, dist_coeffs):
    # Load the image
    image = cv2.imread(image_path)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    # cv2.imshow("Undistorted Image", undistorted_image)
    # cv2.waitKey(0)
    


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

    flat_corners = np.concatenate(corners).tolist()
    flat_corners_array = np.array(flat_corners)

    
    finalCorners = [tuple(flat_corners_array[0][0]),tuple(flat_corners_array[1][3]), tuple(flat_corners_array[2][2]), tuple(flat_corners_array[3][1])]
    
    
    

    # Return the detected markers' information
    return corners, ids, image_markers


def generate_top_down_view(image, extremeCorners):
    
    extremeCorners = sorted(extremeCorners, key=lambda x: (x[1], x[0]))
    print(extremeCorners)
    # Extract the ordered corners
    tl, tr, br, bl = extremeCorners
    

    # compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))


    # Visualization (Optional)
    cv2.imshow('Original Image', image)
    cv2.imshow('Top-Down View', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def calibration():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_dir = os.path.join(script_dir, 'chess')
    image_path = os.path.join(script_dir, 'poolTable.jpg')

    # Create a list of file paths in the calibration directory
    calibration_images = [os.path.join(calibration_dir, file_name) for file_name in os.listdir(calibration_dir) if file_name.endswith(".jpg")]

    # Chessboard size (number of inner corners)
    chessboard_size = (7,7)
    
    # Calibrate the camera and obtain camera matrix and distortion coefficients
    camera_matrix, dist_coeffs = calibrate_camera(calibration_dir, chessboard_size)

    corners, ids, markersDetectedImage = detect_aruco_markers(image_path, camera_matrix, dist_coeffs)
    # print(corners)
    # print()
    # print(extremeCorners)
    # print("Corners: ", corners[1][0])
    finalCorners = [()]*4
    if ids is not None and len(ids) == 4:
        print("Detected 4 ArUco markers:")
        for i in range(4):
            
            print(f"Marker ID {ids[i]} - Corners: {corners[i]}")
            
            if ids[i]==0:
                
                finalCorners[i]=tuple(corners[i][0][0])
            elif ids[i] == 1:
                finalCorners[i]=tuple(corners[i][0][3])
            elif ids[i] == 2:
                finalCorners[i]=tuple(corners[i][0][2])
            elif ids[i] == 3:
                
                finalCorners[i]=tuple(corners[i][0][1])
            
    else:
        print("Could not detect 4 ArUco markers in the image.")
    
    finalCorners = sorted(finalCorners, key=lambda x: (x[1], x[0]))
    
    generate_top_down_view(markersDetectedImage, finalCorners)

    return camera_matrix, dist_coeffs, finalCorners

if __name__ == "__main__":
    calibration()