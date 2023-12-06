import numpy as np
import cv2 as cv
import os
import sys
'''
This code lets you click on a pixel and diplays its values on the image
'''
directoryPath = r'C:\Users\anmol\Documents\UBC\MECH 4\MECH 423\Labs\Final Project\Mech423\Images';

#Load pool table & pool balls
#imgPoolTable = cv.imread(os.path.join(directoryPath,'American-style_pool_table_diagram_(empty).png'))
imgNormal = cv.imread(os.path.join(directoryPath,'American-style_pool_table_diagram_(empty).png'))
img = cv.cvtColor(imgNormal,cv.COLOR_BGR2HSV)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        pixel_value = img[y, x]
        pixel_value_str = f"Pixel Value: {pixel_value}"
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, pixel_value_str, (x, y), font, 1, (255, 255, 255), 2, cv.LINE_AA)

# Create a black image, a window and bind the function to window
cv.namedWindow('image',cv.WINDOW_KEEPRATIO)
#cv.resizeWindow('image', 1000, 1000)
cv.setMouseCallback('image',draw_circle)

while(1):
    cv.imshow('image',img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()