from genericpath import exists
import time
import cv2
from cv2 import imshow
from vehicleDetector import Detector
from centroid_tracker import CentroidTracker
from collections import OrderedDict
from TrackableObject import calculateSpeeds


ip = 0

source = "ds.mp4"

dete = Detector()

ct = CentroidTracker(3)

ontracking = {}
speeds = {}
overspeed = []


if source:
    cap = cv2.VideoCapture(source)
else:

    cap = cv2.VideoCapture(ip)

frame_count = 0
tt_opencvDnn = 0
frameTime= 1/25

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    frame_count += 1
    frameCopy = frame.copy()
    t = time.time()
    outOpencvDnn, bboxesCid = dete.detect(frame)
    tt_opencvDnn += time.time() - t
    fpsOpencvDnn = frame_count / tt_opencvDnn

    label = "OpenCV DNN device = {} FPS : {:.2f}".format( 1 , fpsOpencvDnn)
    cv2.putText(
        outOpencvDnn,
        label,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    objects = ct.update(bboxesCid)
    for (objectID, data) in objects.items():
        print(objectID, type(objectID))
        centroid = data[0]
        rect = data[1]
        if objectID not in ontracking.keys():
            ontracking[objectID] = [centroid]
        else :
            ontracking[objectID].append(centroid)
    speeds = calculateSpeeds(speeds, ontracking, frameTime)

    for (objectID, speed) in speeds.items() : 

        if objectID not in objects.keys():
            continue
        else:
            boxCid =  objects[objectID]

        tl = 3 or round(0.002 * (outOpencvDnn.shape[0] + outOpencvDnn.shape[1]) / 2) + 1  # line/font thickness
        
        tf = max(tl - 1, 1)  # font thickness
        avgSpeed =  sum(speed)/len(speed)

        cv2.putText(outOpencvDnn, "speed = {s:.1f}".format(s =avgSpeed), (boxCid[0][0], boxCid[0][1] - 2) , 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
      
      
       # check speed limit base on class 

        if boxCid[1][4] == 0 and objectID not in overspeed:
            objectImage = frameCopy[boxCid[1][1]:boxCid[1][3],boxCid[1][0]:boxCid[1][2]]
            imshow("ambulance no speed limit ID: {x}".format(x= objectID), objectImage)
            overspeed.append(objectID )

        elif boxCid[1][4] == 1 and objectID not in overspeed:
            if avgSpeed > 70 and len(speed)>10:
                objectImage =  frameCopy[boxCid[1][1]:boxCid[1][3],boxCid[1][0]:boxCid[1][2]]
                imshow("Bus ID: overspeed {x}".format(x= objectID), objectImage)
                overspeed.append(objectID )

        elif boxCid[1][4] == 2 and objectID not in overspeed:
            if avgSpeed > 120 and len(speed)>10:
                objectImage =  frameCopy[boxCid[1][1]:boxCid[1][3],boxCid[1][0]:boxCid[1][2]]
                imshow("car ID: overspeed {x}".format(x= objectID), objectImage)
                overspeed.append(objectID )

        elif boxCid[1][4] == 3 and objectID not in overspeed:
            if avgSpeed > 80 and len(speed)>10:
                objectImage =  frameCopy[boxCid[1][1]:boxCid[1][3],boxCid[1][0]:boxCid[1][2]]
                imshow("motorcycle ID: overspeed {x}".format(x= objectID), objectImage)
                overspeed.append(objectID )

        elif boxCid[1][4] == 4 and objectID not in overspeed:

            if avgSpeed > 60 and len(speed)>10:
                objectImage =  frameCopy[boxCid[1][1]:boxCid[1][3],boxCid[1][0]:boxCid[1][2]]
                imshow("trank ID: overspeed {x}".format(x= objectID), objectImage)
                overspeed.append(objectID )
    
    time.sleep(0.05)
    cv2.imshow("Frame", outOpencvDnn)

    if frame_count == 1:
        tt_opencvDnn = 0

    k = cv2.waitKey(5)
    if k == 27:
        break
t2 = time.time() 
cv2.destroyAllWindows()