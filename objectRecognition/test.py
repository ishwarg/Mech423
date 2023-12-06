import calibration as cc
import cueAngle as ca


camera_matrix, dist_coeffs, finalCorners, topDown = cc.initialCalibration()
angle = ca.determineAngle(topDown, camera_matrix, dist_coeffs)
print(angle)
