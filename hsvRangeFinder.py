# Sourced from https://medium.com/programming-fever/how-to-find-hsv-range-of-an-object-for-computer-vision-applications-254a8eb039fc

#finding hsv range of target object(pen)
import cv2
import numpy as np
import time
import argparse
import ffmpeg
import os
from tkinter.filedialog import askopenfile

# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

# Initializing the webcam feed.
# parser = argparse.ArgumentParser(description='This sample demonstrates the camshift algorithm. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# movie = askopenfile(title='Select Folder Containing Arm Data')
# # parser.add_argument('movie', type=str, help='path to movie file')
# name, ext = os.path.splitext(movie.name)
# print(ext)
# if ext != ".mp4":
#     target_name = name + ".mp4"
#     ffmpeg.input(movie.name).output(target_name).run()
#     print("Finished converting {}".format(movie.name))
# else:
#     target_name = movie.name
target_name = "/Users/alex/Library/Mobile Documents/com~apple~CloudDocs/School/Masters/BME_207/Term Project/20221202_Initial_Full_Data_Run/20221202_FullDataVideo.mp4"
print("Analyzing {}".format(target_name))
cap = cv2.VideoCapture(target_name)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framerate = int(cap.get(cv2.CAP_PROP_FPS))
start_delay_s = 25
print((framerate))
cap.set(1, 7561)#int(start_delay_s*framerate)/length)
cap.set(3,1280)
cap.set(4,720)
# take first frame of the video
# ret,frame = cap.read()

# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
# cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
 
while True:
    time.sleep(0.5)
    # Start reading the webcam feed frame by frame.
    cap.set(1, 8202)
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally (Not required)
    frame = cv2.flip( frame, 1 ) 
    
    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get the new values of the trackbar in real time as the user changes 
    # them 
    # 25.0, 100.0,200.0)), np.array((50.0,200.0,255.0)
    l_h = 155.0#cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = 75.0#cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = 225.0#cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = 200.0#cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = 150.0#cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = 255.0#cv2.getTrackbarPos("U - V", "Trackbars")
 
    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    # Filter the image and get the binary mask, where white represents 
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
 
    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Converting the binary mask to 3 channel image, this is just so 
    # we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3,hsv,res))
    
    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars',cv2.resize(mask,None,fx=0.4,fy=0.4))
    
    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    # If the user presses `s` then print this array.
    if key == ord('s'):
        
        thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(thearray)
        
        # Also save this array as penval.npy
        np.save('hsv_value',thearray)
        break
    
# Release the camera & destroy the windows.    
cap.release()
cv2.destroyAllWindows()