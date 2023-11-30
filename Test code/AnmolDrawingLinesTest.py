import cv2 as cv
import numpy as np
import sys
import os


directory_path = r'C:\Users\anmol\OneDrive\Pictures\Checkerboard'
file_name = 'chessboard.png'

file_path = os.path.join(directory_path, file_name)
img = cv.imread(file_path)

if img is None:
    sys.exit("Could not read image")

#Draw a diagnal blue line
height, width, _ = img.shape

desired_width = 1920
desired_height = 1080/2

cv.line(img,(0,0),(int(height/2),int(width/2)),(255,0,0),5)

resized_img = cv.resize(img, None, fx = desired_height/height, fy=desired_height/height)
cv.imshow("Display window", resized_img)

k=cv.waitKey(0)

