import calibration as cc
import cueAngle as ca
import ballDetection as bd
import os
import cv2

camera_matrix, dist_coeffs, finalCorners, topDown = cc.initialCalibration()
# angle = ca.determineAngle(topDown, camera_matrix, dist_coeffs)
# print(angle)


script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'IMG_0113.jpg')
image =cv2.imread(image_path)

backgroundThreshold = bd.GenerateBackgroundThresholds(image, 10)
print(backgroundThreshold)