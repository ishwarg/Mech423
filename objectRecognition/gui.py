import tkinter as tk
import cv2
from PIL import Image, ImageTk
from PoolTableConstants import *
import numpy as np
import os
import calibration as cc
import ballDetection as bd


class MyGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("GUI with Text Boxes and Image Display")

        self.backgroundThresholdSelect = False
        self.backgroundThreshold = [None]*2
        self.backgroundThresholdsCalibrated = False

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

        # Create button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit_values)
        self.submit_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.backgroundColour_button = tk.Button(master, text="Background Colour", command=self.getBackgroundColour)
        self.backgroundColour_button.grid(row=3, column=2, columnspan=2, pady=10)


        # Image display area
        self.image_label = tk.Label(master)
        self.image_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Set up the video capture (replace '0' with your camera index or file path)
        self.cap = cv2.VideoCapture(1)

        # Call the update method after 100 milliseconds
        self.update()

    def getBackgroundColour(self):
        self.backgroundThresholdSelect = True
        


    def submit_values(self):
        # Retrieve values from text boxes and store them in variables
        value1 = self.textbox1.get()
        value2 = self.textbox2.get()
        value3 = self.textbox3.get()

        # Print the values (you can replace this with your desired logic)
        print("Object Ball ID:", value1)
        print("Cue Ball ID:", value2)
        print("Pocket ID:", value3)

    def update(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
       
        else:
            try:
                corners, ids, image_markers=cc.detect_aruco_markers(frame, camera_matrix, dist_coeffs)
                finalCorners = [(None)]*4

                if ids is not None and len(ids) == 4:
                    #print("Detected 4 ArUco markers:")
                    for i in range(4):
                        
                        #print(f"Marker ID {ids[i]} - Corners: {corners[i]}")
                        
                        if ids[i]==0:
                            finalCorners[int(ids[i])]=tuple(corners[i][0][0])
                        elif ids[i] == 1:
                            finalCorners[int(ids[i])]=tuple(corners[i][0][0])
                        elif ids[i] == 2:
                            finalCorners[int(ids[i])]=tuple(corners[i][0][0])
                        elif ids[i] == 3:
                            
                            finalCorners[int(ids[i])]=tuple(corners[i][0][0])
                        
                else:
                    print("Could not detect 4 ArUco markers in the image.")
                warped = cc.generate_top_down_view(image_markers, finalCorners, MAX_WIDTH, MAX_HEIGHT)
                if self.backgroundThresholdSelect:
                    self.backgroundThreshold = bd.GenerateBackgroundThresholds(warped, 100)
                    self.backgroundThresholdSelect = False
                    self.backgroundThresholdsCalibrated = True
                    print(self.backgroundThreshold)
                if self.backgroundThresholdsCalibrated:
                    pass
                rgb_frame = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (1000, 500))  

                # Convert the frame to PhotoImage format
                photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

                # Display the frame
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            except Exception as e:
                    print(f"Error in detect_aruco_markers: {e}")
            

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
