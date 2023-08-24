# Code starting point from:
# https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html
# https://github.com/spmallick/learnopencv/tree/master/MultiObjectTracker
import numpy as np
import cv2 as cv
import argparse
from random import randint
import pandas
import time
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
cap = cv.VideoCapture(target_name)
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
framerate = int(cap.get(cv.CAP_PROP_FPS))
start_delay_s = 25
print((framerate))
startTime = 32000 # Desired analysis start time in ms (32000)
cap.set(0, startTime) # int(start_delay_s*framerate)/length)
cap.set(3,1280)
cap.set(4,720)
# take first frame of the video
ret,frame = cap.read()
## Select regions of interest
bboxes = [(266, 628, 56, 51), (560, 620, 47, 33), (997, 563, 40, 39)]#[]
colors = [] 
colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

  
print('Selected bounding boxes {}'.format(bboxes))

l_h = 155.0
l_s = 75.0
l_v = 150.0
u_h = 175.0
u_s = 150.0
u_v = 255.0

# Set the lower and upper HSV range according to the value selected
# by the trackbar
lower_range = np.array([l_h, l_s, l_v])
upper_range = np.array([u_h, u_s, u_v])


# set up the ROI for tracking
# Mask only the green color of the joint markers
x, y, w, h, track_window, roi, hsv_roi, mask, roi_hist, term_crit = [], [], [], [], [], [], [], [], [], []
tracking_coordinates = []#np.zeros((len(bboxes),1, 2))
for i in range(len(bboxes)):
    tracking_coordinates.append(np.zeros((1, 2)))
    x.append(bboxes[i][0])
    y.append(bboxes[i][1])
    w.append(bboxes[i][2])
    h.append(bboxes[i][3])
    track_window.append((x[i], y[i], w[i], h[i]))
    roi.append(frame[y[i]:y[i]+h[i], x[i]:x[i]+w[i]])
    hsv_roi.append(cv.cvtColor(roi[i], cv.COLOR_BGR2HSV))
    mask.append(cv.inRange(hsv_roi[i], lower_range, upper_range))
    # roi_hist.append(cv.calcHist([hsv_roi[i]],[0, 1, 2],mask[i],[int(u_h-l_h), int(u_s-l_s), int(u_v-l_v)],[l_h, u_h, l_s, u_s, l_v, u_v]))
    roi_hist.append(cv.calcHist([hsv_roi[i]],[0],mask[i],[int(u_h-l_h)],[l_h, u_h]))
    cv.normalize(roi_hist[i],roi_hist[i],0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 5 )
angles = pandas.DataFrame(data = {"Frame": [], "Angle": []})
frame_num = 0
error_flag = False
while(1):
    frame_num = frame_num + 1
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        drawPts = np.empty((1,2))
        for i in range(len(bboxes)):
            # dst = cv.calcBackProject([hsv],[0, 1, 2],roi_hist[i],[l_h, u_h, l_s, u_s, l_v, u_v],1)
            dst = cv.calcBackProject([hsv],[0],roi_hist[i],[l_h, u_h],1)
            # cv.imshow('DST Mask',mask[i])
            # k = cv.waitKey(30) & 0xff
            # apply camshift to get the new location
            ret, track_window[i] = cv.meanShift(dst, track_window[i], term_crit)
            # convert image to grayscale image
            rectPts = track_window[i]
            #update histogram with latest point
            hsv_roi[i] = hsv[rectPts[1]:rectPts[1]+rectPts[3], rectPts[0]:rectPts[0]+rectPts[2]]
            mask[i] = cv.inRange(hsv_roi[i], lower_range, upper_range)
            # roi_hist.append(cv.calcHist([hsv_roi[i]],[0, 1, 2],mask[i],[int(u_h-l_h), int(u_s-l_s), int(u_v-l_v)],[l_h, u_h, l_s, u_s, l_v, u_v]))
            roi_hist[i] = (cv.calcHist([hsv_roi[i]],[0],mask[i],[180],[l_h, u_h]))
            # convert the grayscale image to binary image
            grayret,thresh = cv.threshold( mask[i],127,255,0)
            # # calculate moments of binary image
            M = cv.moments(thresh)
            # # calculate x,y coordinate of center of color blob
            try:
                coord_array = np.array([[int(rectPts[0]) + int(M["m10"] / M["m00"]), int(rectPts[1]) + int(M["m01"] / M["m00"])]])
            except:
                print("Encountered Error at frame number {}, {}ms while tracking point {} Bounding Rectangle is {}. Moments are {}.".format(\
                         cap.get(cv.CAP_PROP_POS_FRAMES), cap.get(cv.CAP_PROP_POS_MSEC), i, rectPts, M))
                # coord_array = last_coords[i]
                mask_error = dst#cv.cvtColor(dst, cv.COLOR_HSV2BGR)
                error_flag = True
                cv.imshow('Error Mask',mask_error)
                k = cv.waitKey(30) & 0xff
                time.sleep(1000)
            tracking_coordinates[i] = np.append(tracking_coordinates[i], coord_array, 0)
            # Add coordinates to our array of coordinates
            drawPts = np.append(drawPts, coord_array, 0)
        # Draw pts on image
        if error_flag == False:
            drawPts = np.int0(drawPts)
            drawPts = drawPts[1:]
            img2 = cv.polylines(frame,[drawPts],False, colors[0],2)
            # Calculate joint angle - this only works as expected if the points are selected shoulder --> elbow --> wrist
            ba = drawPts[0] - drawPts[1]
            bc = drawPts[2] - drawPts[1]
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.degrees(np.arccos(cosine_angle))
            angles.loc[len(angles)] = [frame_num, angle]
            cv.imshow('img2',img2)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            error_flag = False
    else:
        break
# Export to CSV file
angles.to_csv('armData.csv')