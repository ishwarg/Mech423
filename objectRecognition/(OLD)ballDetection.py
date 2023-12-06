import cv2
import numpy as np
import matplotlib.pyplot as plt
import calibration as cc
import os




height, width = 280, 560 # size of output image (few functions use it)
# hsv colors of the snooker table
lower = np.array([60, 200,150]) 
upper = np.array([70, 255,240]) # HSV of snooker green: (60-70, 200-255, 150-240) 

# Function to create 2d table image
# Input: None
# Output: img
def create_table():
    
    # new generated img 
    img = np.zeros((height,width,3), dtype=np.uint8) # create 2D table image 
    img[:, :] = [0, 180, 10] # setting RGB colors to green pool table color, (0,180,10)=certain green
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          #For compatibility with pyplot
    
    ''' Semi-circle on snooker table
    # create circle in the right size
    cv2.circle(img, (int(width/2),int(height/5)), # center of circle
               int((width/3)/2), # radius
               (50,255,50)) # color
    
    # delete half of circle by coloring in green color
    img[int(height/5):height,0:width] = [0, 180, 10] 
    '''
    # create line
    cv2.line(img,(int(width/5),0),(int(width/5),height),(50,255,50)) 
    
    return img

#Function to draw border and holes
#Input: 2d table
#Output: 2dTable with holes and borders
def draw_holes(input_img, color3 = (200,140,0)):
        
    color = (190, 190, 190) # gray color
    color2 = (120, 120, 120) #  gray color, for circles (holes) on generated img

    img = input_img.copy() # make a copy of input image
    
    # borders 
    cv2.line(img,(0,0),(width,0),color3,3) # top
    cv2.line(img,(0,height),(width,height),color3,3) # bot
    cv2.line(img,(0,0),(0,height),color3,3) # left
    cv2.line(img,(width,0),(width,height),color3,3) # right
    
    # adding circles to represent holes on table
    cv2.circle(img, (0, 0), 11,color, -1) # top right
    cv2.circle(img, (width,0), 11, color, -1) # top left
    cv2.circle(img, (0,height), 11, color, -1) # bot left
    cv2.circle(img, (width,height), 11, color, -1) # bot right
    cv2.circle(img, (int(width/2),height), 8, color, -1) # mid bot
    cv2.circle(img, (int(width/2),0), 8, color, -1) # mid top
    
    # adding another, smaller circles to the previous ones
    cv2.circle(img, (0, 0), 9,color2, -1) # top right
    cv2.circle(img, (width,0), 9, color2, -1) # top left
    cv2.circle(img, (0,height), 9, color2, -1) # bot left
    cv2.circle(img, (width,height), 9, color2, -1) # bot right
    cv2.circle(img, (int(width/2),height), 6, color2, -1) # mid right
    cv2.circle(img, (int(width/2),0), 6, color2, -1) # mid left
    
    return img

#Function to find balls on physical pool table and draw them
#Input: Ball contours, pooltable as background, ball radius size of lines
#Output image with balls
def draw_balls(ctrs,background = create_table(), radius=7, size = -1, img = 0):

    K = np.ones((3,3),np.uint8) # filter
    font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 30, 3)
    
    final = background.copy() # canvas
    mask = np.zeros((height, width),np.uint8) # empty image, same size as 2d generated final output
    
    
    for x in range(len(ctrs)): # for all contours
        
        # find center of contour
        M = cv2.moments(ctrs[x])
        cX = int(M['m10']/M['m00']) # X pos of contour center
        cY = int(M['m01']/M['m00']) # Y pos
        
        # find color average inside contour
        mask[...]=0 # reset the mask for every ball 
        cv2.drawContours(mask,ctrs,x,255,-1) # draws mask for each contour
        mask =  cv2.erode(mask,K,iterations = 3) # erode mask several times to filter green color around balls contours
        
        
        # balls design:
        
        
        # circle to represent snooker ball
        final = cv2.circle(final, # img to draw on
                           (cX,cY), # position on img
                           radius, # radius of circle - size of drawn snooker ball
                           cv2.mean(img,mask), # color mean of each contour-color of each ball (src_img=transformed img)
                           size) # -1 to fill ball with color
        
        # add black color around the drawn ball (for cosmetics)
        final = cv2.circle(final, (cX,cY), radius, 0, 1) 
        final = cv2.putText(final,
                    "Location",
                    (cX,cY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)
         
        

        
    return final

'''
gets contours and filters them by their size and shape, then returns filtered contours.

* input:
       - ctrs: contours
       - min_s: min area size of accepted contour
       - max_s: max area size of accepted contour
       - alpha: parameter to find wanted shape

* output: 
        contours
'''
#alpha is based on ball distortion
def filter_ctrs(ctrs, min_s = 90, max_s = 358, alpha = 3.445):  
    
    filtered_ctrs = [] # list for filtered contours
    
    for x in range(len(ctrs)): # for all contours
        
        rot_rect = cv2.minAreaRect(ctrs[x]) # area of rectangle around contour
        w = rot_rect[1][0] # width of rectangle
        h = rot_rect[1][1] # height
        area = cv2.contourArea(ctrs[x]) # contour area 

        
        if (h*alpha<w) or (w*alpha<h): # if the contour isnt the size of a snooker ball
            continue # do nothing
            
        if (area < min_s) or (area > max_s): # if the contour area is too big/small
            continue # do nothing 

        # if it failed previous statements then it is most likely a ball
        filtered_ctrs.append(ctrs[x]) # add contour to filtered cntrs list

        
    return filtered_ctrs # returns filtere contours

# This function determines the x-y coords of all the balls
# This function also filters contours
# input: contours, mminimum and maximum radii
# Output: x-y coordinates of each ball
def FindBalls(ctrs, min_r = 25, max_r = 50):

    l = [] # list for filtered contours
    
    for c in ctrs: # for all contours
        
        M = cv2.moments(c)

        # If area smaller than a threshold filter it out
        if M["m00"]<np.pi*min_r**2:
            continue

        # If contour has straight lines filter it out
        lX=[x for [[x, _]] in c]
        lY=[y for [[_, y]] in c]
        
        if np.corrcoef(lX, lY)[0, 1]**2 > 0.75:
            [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            cv2.line(
                newframe,
                tuple(map(int, (x+vx*-1920, y+vy*-1920))),
                tuple(map(int, (x+vx*1920, y+vy*1920))),
                (255, 255, 255),
                15
            )
            continue

        # If contour is circular include it in final list
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        stdRadius = np.std([((x-ix)**2 + (y-iy)**2)**0.5 for ix, iy in zip(lX, lY)])

        if stdRadius < 10:
           l+=[(x, y)]

    return l
'''
gets a frame (of snooker table), applies several methods to detect the balls and returns 2D top view with drawn, colored balls


* input:
        src: image, frame from snooker video

* output: 
        image: 2D top view
'''
def find_balls(src):
    final = create_table()
    '''Assume img is already warped & calibrated
    # warp perspective
    matrix = cv2.getPerspectiveTransform(pts1,pts2) # getting perspective by 4 points of each image
    transformed = cv2.warpPerspective(src, matrix, (width,height)) # warps perpective to new image
    '''
    transformed = src

    # apply blur
    transformed_blur = cv2.GaussianBlur(transformed,(5,5),cv2.BORDER_DEFAULT) # blur applied
    blur_RGB = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2RGB) # rgb version

    # mask
    hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV) # convert to hsv
    mask = cv2.inRange(hsv, lower, upper) # table's mask

    # filter mask
    kernel = np.ones((5,5),np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # dilate->erode
    
    # apply threshold
    ret,mask_inv = cv2.threshold(mask_closing,5,255,cv2.THRESH_BINARY_INV) # apply threshold
       
    # create image with masked objects on table 
    masked_objects = cv2.bitwise_and(transformed,transformed, mask=mask_inv) # masked image

    # find contours and filter them
    ctrs, hierarchy = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours
    ctrs = filter_ctrs(ctrs) # filter contours by sizes and shapes

    # draw table+balls
    final = draw_balls(ctrs,radius=8,img=transformed) # draw all found contours  
    final = draw_holes(final) # draw holes
    
    return final

'''
gets image (of snooker table), applies several methods to detect the balls and returns 2D top view with drawn, colored balls


* input:
        ctrs: contours
        input_img: image that the contours are taken from

* output: 
        image, black image with large colored circle for each contour

'''
#This function is redundant use draw balls instead
def find_ctrs_color(ctrs, input_img):

    K = np.ones((3,3),np.uint8) # filter
    output = input_img.copy() #np.zeros(input_img.shape,np.uint8) # empty img
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) # gray version
    mask = np.zeros(gray.shape,np.uint8) # empty mask

    for i in range(len(ctrs)): # for all contours
        
        # find center of contour
        M = cv2.moments(ctrs[i])
        cX = int(M['m10']/M['m00']) # X pos of contour center
        cY = int(M['m01']/M['m00']) # Y pos
    
        mask[...]=0 # reset the mask for every ball 
    
        cv2.drawContours(mask,ctrs,i,255,-1) # draws the mask of current contour (every ball is getting masked each iteration)

        mask =  cv2.erode(mask,K,iterations=3) # erode mask to filter green color around the balls contours
        
        output = cv2.circle(output, # img to draw on
                         (cX,cY), # position on img
                         20, # radius of circle - size of drawn snooker ball
                         cv2.mean(input_img,mask), # color mean of each contour-color of each ball (src_img=transformed img)
                         -1) # -1 to fill ball with color
    return output

if __name__ == "__main__":
    camera_matrix, dist_coeffs, finalCorners = cc.initialCalibration()
    targetCorners = np.float32([ [0,0],[width,0],[0,height],[width,height] ]) # 4 corners points of OUTPUT image
    #name = 'P6_Snooker.mp4' #Test video

    #Test image flatten based on calibration data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'poolTable.jpg')
    image = cv2.imread(image_path)
    image = cv2.undistort(image, camera_matrix, dist_coeffs)
    
    M = cv2.getPerspectiveTransform(finalCorners,targetCorners)
    image_warped = cv2.warpPerspective(image, M, (width, height))

    find_balls(image_warped)

