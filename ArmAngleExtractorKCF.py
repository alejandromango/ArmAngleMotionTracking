# Code starting point from:
# https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html
# https://github.com/spmallick/learnopencv/tree/master/MultiObjectTracker
import numpy as np
import cv2 as cv
import argparse
from random import randint
import pandas
import time

print(cv.TrackerCSRT.create())
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
#[(266, 628, 56, 51), (560, 620, 47, 33), (997, 563, 40, 39)]
bboxes = [(266, 628, 56, 51), (560, 620, 47, 33), (997, 563, 40, 39)]#[]
colors = [] 
colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
# while True:
#     # draw bounding boxes over objects
#     # selectROI's default behaviour is to draw box starting from the center
#     # when fromCenter is set to false, you can draw box starting from top left corner
#     bbox = cv.selectROI('MultiTracker', frame)
#     bboxes.append(bbox)
#     colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
#     print("Press q to quit selecting boxes and start tracking")
#     print("Press any other key to select next object")
#     k = cv.waitKey(0) & 0xFF
#     if (k == 113):  # q is pressed
#       break
  
print('Selected bounding boxes {}'.format(bboxes))

l_h = 10.0#cv2.getTrackbarPos("L - H", "Trackbars")
l_s = 75.0#cv2.getTrackbarPos("L - S", "Trackbars")
l_v = 225.0#cv2.getTrackbarPos("L - V", "Trackbars")
u_h = 175.0#cv2.getTrackbarPos("U - H", "Trackbars")
u_s = 150.0#cv2.getTrackbarPos("U - S", "Trackbars")
u_v = 255.0#cv2.getTrackbarPos("U - V", "Trackbars")

# Set the lower and upper HSV range according to the value selected
# by the trackbar
lower_range = np.array([l_h, l_s, l_v])
upper_range = np.array([u_h, u_s, u_v])


# set up the ROI for tracking
# Mask only the green color of the joint markers
track_window, roi, hsv_roi, mask = [], [], [], []
tracking_coordinates = []
trackers = []
for i in range(len(bboxes)):
    tracking_coordinates.append(np.zeros((1, 2)))
    trackers.append(cv.legacy.TrackerMedianFlow_create())
    track_window.append(bboxes[i])
    ok = trackers[i].init(frame, bboxes[i])
    test = track_window
    roi.append(frame[track_window[i][1]:track_window[i][1]+track_window[i][3], track_window[i][0]:track_window[i][0]+track_window[i][2]])
    hsv_roi.append(cv.cvtColor(roi[i], cv.COLOR_BGR2HSV))
    mask.append(cv.inRange(hsv_roi[i], lower_range, upper_range))
    if not ok:
        print('[ERROR] tracker not initialized')
        exit()
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
angles = pandas.DataFrame(data = {"Frame": [], "Angle": []})
frame_num = 0
error_flag = False
while(1):
    frame_num = frame_num + 1
    ret, frame = cap.read()
    if (cap.get(cv.CAP_PROP_POS_FRAMES) % 1000 == 0):
        print("Alanayzed {} frames of {}".format(cap.get(cv.CAP_PROP_POS_FRAMES), cap.get(cv.CAP_PROP_FRAME_COUNT)))
    if (ret == True) & (cap.get(cv.CAP_PROP_POS_FRAMES) < 24800):
        drawPts = np.empty((1,2))
        for i in range(len(bboxes)):
            # apply algorithm to get the new location
            ret, track_window[i] = trackers[i].update(frame)
            #update histogram with latest point
            try:
                track_window[i] = (int(track_window[i][0]), int(track_window[i][1]), int(track_window[i][2]), int(track_window[i][3]))
                roi[i] = frame[track_window[i][1]:track_window[i][1]+track_window[i][3], track_window[i][0]:track_window[i][0]+track_window[i][2]]
                hsv_roi[i] = cv.cvtColor(roi[i], cv.COLOR_BGR2HSV)
                mask[i] = cv.inRange(hsv_roi[i], lower_range, upper_range)
                # convert the grayscale image to binary image
                grayret,thresh = cv.threshold( mask[i],127,255,0)
                # # calculate moments of binary image
                M = cv.moments(thresh)
                # # calculate x,y coordinate of center of color blob

                coord_array = np.array([[int(track_window[i][0]) + int(M["m10"] / M["m00"]), int(track_window[i][1]) + int(M["m01"] / M["m00"])]])
            except:
                print("Tracking of point {} failed, please manually locate it to continue tracking at frame number {}".format(i, cap.get(cv.CAP_PROP_POS_FRAMES)))
                failure_bbox = cv.selectROI("Select ROI number {}".format(i), frame)
                trackers[i].init(frame, failure_bbox)
                track_window[i] = failure_bbox
                roi[i] = frame[track_window[i][1]:track_window[i][1]+track_window[i][3], track_window[i][0]:track_window[i][0]+track_window[i][2]]
                hsv_roi[i] = cv.cvtColor(roi[i], cv.COLOR_BGR2HSV)

                l_h = 10.0#cv2.getTrackbarPos("L - H", "Trackbars")
                l_s = 75.0#cv2.getTrackbarPos("L - S", "Trackbars")
                l_v = 100.0#cv2.getTrackbarPos("L - V", "Trackbars")
                u_h = 200.0#cv2.getTrackbarPos("U - H", "Trackbars")
                u_s = 255.0#cv2.getTrackbarPos("U - S", "Trackbars")
                u_v = 255.0#cv2.getTrackbarPos("U - V", "Trackbars")

                # Set the lower and upper HSV range according to the value selected
                # by the trackbar
                lower_range = np.array([l_h, l_s, l_v])
                upper_range = np.array([u_h, u_s, u_v])

                #Difficulties at frames 4149, 7561

                mask[i] = cv.inRange(hsv_roi[i], lower_range, upper_range)
                # convert the grayscale image to binary image
                grayret,thresh = cv.threshold( mask[i],127,255,0)
                # # calculate moments of binary image
                M = cv.moments(thresh)
                stacked = np.hstack((roi[i],hsv_roi[i],cv.cvtColor(mask[i], cv.COLOR_GRAY2BGR)))
                cv.imshow('Failure ROI',stacked)
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
                # time.sleep(1000)
                coord_array = np.array([[int(track_window[i][0]) + int(M["m10"] / M["m00"]), int(track_window[i][1]) + int(M["m01"] / M["m00"])]])
                print("recovered from failure")

            tracking_coordinates[i] = np.append(tracking_coordinates[i], coord_array, 0)
            # Add coordinates to our array of coordinates
            drawPts = np.append(drawPts, coord_array, 0)
        # Draw pts on image
        drawPts = np.int0(drawPts)
        drawPts = drawPts[1:]
        img2 = cv.polylines(frame,[drawPts],False, colors[0],2)
        # Calculate joint angle - this only works as expected if the points are selected shoulder --> elbow --> wrist
        ba = drawPts[0] - drawPts[1]
        bc = drawPts[2] - drawPts[1]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))
        angles.loc[len(angles)] = [frame_num, angle]
        # cv.imshow('img2',img2)
        # k = cv.waitKey(30) & 0xff
        # if k == 27:
        #     break

    else:
        break
# Export to CSV file
angles.to_csv('MotionCaptureAngleData.csv')