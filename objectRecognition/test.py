import calibration as cc
import cueAngle as ca
import ballDetection as bd

import os
import cv2
import numpy as np

#camera_matrix, dist_coeffs, finalCorners, warped = cc.initialCalibration()

# print(camera_matrix, dist_coeffs)

camera_matrix, dist_coeffs = cc.video_calibration((7,7))
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


# script_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(script_dir, 'calibration_data.npz')

# data = np.load(file_path)
# camera_matrix = data['camera_matrix']
# dist_coeffs = data['dist_coeffs']


# script_dir = os.path.dirname(os.path.abspath(__file__))
# image_path = os.path.join(script_dir, 'captured_frame.jpg')
# image =cv2.imread(image_path)

# finalCorners, warped=cc.tableDetection(image, camera_matrix, dist_coeffs)








