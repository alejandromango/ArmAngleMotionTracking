# Code starting point from:
# https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html
# https://github.com/spmallick/learnopencv/tree/master/MultiObjectTracker
import numpy as np
import cv2 as cv
import argparse
from random import randint
import pandas
parser = argparse.ArgumentParser(description='This sample demonstrates the camshift algorithm. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)
# take first frame of the video
ret,frame = cap.read()
## Select regions of interest
bboxes = []
colors = [] 
while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
      break
  
print('Selected bounding boxes {}'.format(bboxes))
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
    mask.append(cv.inRange(hsv_roi[i], np.array((25.0, 100.0,200.0)), np.array((50.0,200.0,255.0))))
    roi_hist.append(cv.calcHist([hsv_roi[i]],[0],mask[i],[180],[0,180]))
    cv.normalize(roi_hist[i],roi_hist[i],0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
angles = pandas.DataFrame(data = {"Frame": [], "Angle": []})
frame_num = 0
while(1):
    frame_num = frame_num + 1
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        drawPts = np.empty((1,2))
        for i in range(len(bboxes)):
            dst = cv.calcBackProject([hsv],[0],roi_hist[i],[0,180],1)
            # apply camshift to get the new location
            ret, track_window[i] = cv.CamShift(dst, track_window[i], term_crit)
            # convert image to grayscale image
            pts = cv.boxPoints(ret)
            rectPts = cv.boundingRect(pts)
            # Isolate just the green color region
            mask_Tracked = cv.inRange(hsv[rectPts[1]:rectPts[1]+rectPts[3], rectPts[0]:rectPts[0]+rectPts[2]], np.array((25.0, 100.0,200.0)), np.array((50.0,200.0,255.0)))
            # convert the grayscale image to binary image
            grayret,thresh = cv.threshold(mask_Tracked,127,255,0)
            # # calculate moments of binary image
            M = cv.moments(thresh)
            # # calculate x,y coordinate of center of color blob
            coord_array = np.array([[int(rectPts[0]) + int(M["m10"] / M["m00"]), int(rectPts[1]) + int(M["m01"] / M["m00"])]])
            tracking_coordinates[i] = np.append(tracking_coordinates[i], coord_array, 0)
            # Add coordinates to our array of coordinates
            drawPts = np.append(drawPts, coord_array, 0)
        # Draw pts on image
        drawPts = np.int0(drawPts)
        drawPts = drawPts[1:]
        img2 = cv.polylines(frame,[drawPts],False, colors[i],2)
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
        break
# Export to CSV file
angles.to_csv('armData.csv')