from datetime import datetime, date
import time
import math
import imutils
import numpy as np
import cv2
import csv


# load the COCO class labels our YOLO model was trained on
# ex. person, bicycle, car
labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# load the YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet('./yolov3.cfg','./yolov3.weights')

# Video Captures
#cap = cv2.VideoCapture("pedestrians.mp4")
#cap = cv2.VideoCapture("people.mp4")
#cap = cv2.VideoCapture("socials.mp4")
cap = cv2.VideoCapture(0)

# to only print the header once in the report log after running program
once = True
# for the update of report log to only occur once per second
oldtime = ""

while(cap.isOpened()):
    # read video feed into image variable
    ret, image = cap.read()

    # grab the dimensions of the frame
    image = imutils.resize(image, width=800)
    (H, W) = image.shape[:2]
    
    # determine only the *output* layer names that we need from YOLO
    # Get the name of all layers of the network.
    ln = net.getLayerNames()
    # Get the index of the output layers.
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    # https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    # processes the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, centroids, and
    # confidences, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
	    # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object
	    # detected was a person and (2) that the minimum
	    # confidence is met
            if confidence > 0.1 and classID == 0:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
		# actually returns the center (x, y)-coordinates of
		# the bounding box followed by the boxes' width and
		# height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
		# and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
		# centroids, and confidences
                boxes.append([x, y, int(width), int(height), centerX, centerY])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)

    a = []
    b = []

    # ensure at least one detection exists
    if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][4], boxes[i][5])
                
                a.append(x)
                b.append(y)

    nsd = []
    centroids = []
    # not safe distance
    for i in range(0,len(a)-1):
        for k in range(1,len(a)):
            if(k==i):
                break
            else:
                # check for violation

                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                # euclidian distance
                # compute for distance between two bounding box
                d = math.sqrt((x_dist * x_dist) + (y_dist * y_dist))

                # violation distance
                if(d <= 500):
                    cv2.line(image, (a[i],b[i]), (a[k],b[k]), (0, 0, 255), 2)
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))

    # if violate, show red bounding box and red text            
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        (cx, cy) = (boxes[i][4], boxes[i][5])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), 2)
        #cv2.putText(image, 'Violation', (x, y - 5), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255), 1)

    # if safe, show greeb bounding box and green text    
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (i in nsd):
                break
            else:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                (cx, cy) = (boxes[i][4], boxes[i][5])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 4, (0, 255, 0), 2)
                #cv2.putText(image, 'Safe', (x, y - 5), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0), 1)
    
    if len(idxs) < 5:
        crowd = 0;
        cd = "LOW"
    elif len(idxs) >= 5 and len(idxs) < 10:
        crowd = 1;
        cd = "MEDIUM"
    else:
        crowd = 2;
        cd = "HIGH"

    # Report Log
    f = open('ReportLog.csv', 'a')
    header = ['Location', 'Crowd Density', 'Number of People ', 'Social Distancing Violations', 'Time', 'Date']
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    data = ["Main gate",cd,len(idxs),len(nsd),current_time,d1]
    writer = csv.writer(f)
    
    if once:
        writer.writerow(header)
        once = False

    if (oldtime != current_time):
        writer.writerow(data)
        oldtime = current_time
        
    if crowd == 0:
        
        cv2.putText(image, "Crowd Density: LOW - "+str(len(idxs)), (10, image.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 3)
    elif crowd == 1:
        cv2.putText(image, "Crowd Density: MEDIUM - "+str(len(idxs)), (10, image.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 3)
    else:
        cv2.putText(image, "Crowd Density: HIGH - "+str(len(idxs)), (10, image.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    
    cv2.imshow("Social Distancing Monitor", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
f.close()
cap.release()
cv2.destroyAllWindows()




