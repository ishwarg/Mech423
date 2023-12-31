import tkinter as tk
import cv2
from PIL import Image, ImageTk
from PoolTableConstants import *
import numpy as np
import os
import calibration as cc
import ballDetection as bd
import cueAngle as ca
import math as m
import traceback
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play



import sys
sys.path.append('/Users/ishwarjotgrewal/Desktop/Mech423')
import shotSelection.physicsModel as pm


BACKGROUND_THRESHOLDS = {
    'upper': np.array([180,230,255]),
    'upperMiddle':np.array([175,180,180]),
    'lowerMiddle': np.array([5,230,255]),
    'lower':np.array([0,180,180])
}



class MyGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("GUI with Text Boxes and Image Display")

        self.cueVector = np.array([0,0])
        self.objectBall = 1
        self.cueBall = 0
        self.pocket = 4
        self.previousCorners = np.array([
		[XOFFSET, 0],
		[MAX_WIDTH -XOFFSET, 0],
		[MAX_WIDTH-XOFFSET, MAX_HEIGHT],
		[XOFFSET, MAX_HEIGHT]], dtype = "int32")
        self.shotTaken = True
        self.BallsIdentified = False
        self.balls = []
       # Create text boxes
        self.textbox1_label = tk.Label(master, text="Object Ball ID:")
        self.textbox1_label.grid(row=0, column=0, padx=5, pady=5)
        self.textbox1 = tk.Entry(master)
        self.textbox1.grid(row=0, column=1, padx=5, pady=5)

        self.textbox2_label = tk.Label(master, text="Cue Ball ID:")
        self.textbox2_label.grid(row=1, column=0, padx=5, pady=5)
        self.textbox2 = tk.Entry(master)
        self.textbox2.grid(row=1, column=1, padx=5, pady=5)

        self.textbox3_label = tk.Label(master, text="Pocket ID:")
        self.textbox3_label.grid(row=2, column=0, padx=5, pady=5)
        self.textbox3 = tk.Entry(master)
        self.textbox3.grid(row=2, column=1, padx=5, pady=5)

        self.textbox4_label = tk.Label(master, text="Angle: ")
        self.textbox4_label.grid(row=2, column=2, padx=5, pady=5)
        self.textbox4 = tk.Label(master, text = "(0,0)")
        self.textbox4.grid(row=2, column=3, padx=5, pady=5)
        

        # Create button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit_values)
        self.submit_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.backgroundColour_button = tk.Button(master, text="Done Shot", command=self.doneShot)
        self.backgroundColour_button.grid(row=3, column=2, columnspan=2, pady=10)


        # Image display area
        self.image_label = tk.Label(master)
        self.image_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Set up the video capture (replace '0' with your camera index or file path)
        self.cap = cv2.VideoCapture(1)
        
        # Call the update method after 100 milliseconds
        self.update()

    def doneShot(self):
        self.shotTaken = True
        self.BallsIdentified = False
        
    

    def submit_values(self):
        # Retrieve values from text boxes and store them in variables
        self.objectBall = int(self.textbox1.get())
        self.cueBall = int(self.textbox2.get())
        self.pocket = int(self.textbox3.get())

        # Print the values (you can replace this with your desired logic)
        print("Object Ball ID:", self.objectBall)
        print("Cue Ball ID:", self.cueBall)
        print("Pocket ID:", self.pocket)
        self.shotTaken = False
        self.BallsIdentified = True

    def update(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
       
        else:
            try:
                warped, self.previousCorners = cc.tableDetection(frame, camera_matrix, dist_coeffs, self.previousCorners)
                
                if not self.BallsIdentified:
                    self.balls = bd.FindandDrawBalls(warped, BACKGROUND_THRESHOLDS)
                    
                else:
                    bd.DrawBalls(self.balls, warped)
                
                balls = self.balls

                if not self.shotTaken:
                    objectBallTraj = pm.ObjectBallTraj(balls,self.objectBall,self.pocket)
                    cueBallTraj = pm.CueBallTraj(balls,self.objectBall,self.cueBall,objectBallTraj[0])
                    pm.DrawTraj(warped,balls[self.objectBall],objectBallTraj) 
                    pm.DrawTraj(warped,balls[self.cueBall],cueBallTraj)  
                    
                    

                    self.cueVector, cueCorners, found =ca.determineAngle(warped, self.cueVector)

                    if found:

                        
                        # print(cueCorners[0][0][0])
                        ca.DrawInfiniteLine(warped, cueCorners[0][0][0], self.cueVector)

                       
                        rounded_array = np.round(self.cueVector, decimals=2)
                        array_string = np.array2string(rounded_array, precision=2, suppress_small=True)
                        self.textbox4.config(text=array_string)
                        magnitude = np.linalg.norm(cueBallTraj)
                        unit_traj = cueBallTraj/magnitude
                        
                        dot_product = np.dot(self.cueVector, unit_traj[0])
                        

                        # Calculate the allowable range based on the tolerance percentage
                        

                        # Check if the dot product is within the allowable range
                        angleCheck = abs(dot_product - 1)<=0.005
                        
                        if angleCheck:
                            print("Hurray!")
                            ding_sound = Sine(1000).to_audio_segment(duration=100).fade_in(5).fade_out(5)
                            play(ding_sound)
                    else: 
                        print("Cue not found!!")
                  
                rgb_frame = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (1000, 500))  

                # Convert the frame to PhotoImage format
                photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

                # Display the frame
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            except Exception as e:
                    traceback_details = traceback.format_exc()
                    print(f"Error in detect_aruco_markers: {e}\n{traceback_details}")
            

        # Call the update method again after 100 milliseconds
        self.master.after(100, self.update)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'calibration_data.npz')

    data = np.load(file_path)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    


    root = tk.Tk()
    my_gui = MyGUI(root)
    root.mainloop()
