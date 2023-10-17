
import cv2
import numpy as np
import time
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects



class YoloDetector():

    def __init__(self, model_name):
        
        self.model = self.load_model(model_name)
        self.classes = self.model.names

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        

    def load_model(self, model_name):
   
        if model_name:
            model = torch.hub.load('/home/frinksserver/Desktop/Abhishek/object_tracking/yolov5', "custom", source='local',path=model_name,force_reload=True)
            
        else:
            # model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
            pass
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        downscale_factor = 2
        if frame is None:
            return
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width,height))
        
        results = self.model(frame)
        
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        
        return labels, cord

    def class_to_label(self, x):
   
        return self.classes[int(x)]


    def plot_boxes(self, results, frame, height, width, confidence=0.3):
    
        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height
        
    
    
        for i in range(n):
            row = cord[i]
            
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        
                confidence = float(row[4].item())
                feature = self.class_to_label(labels[i])

                detections.append([x1, y1, x2, y2])
                
        
        return frame, detections


def main():
    cap = cv2.VideoCapture('video2.avi')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)
    fps = int(cap.get(5))
    print(fps)


    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    size = (frame_width,frame_height)

    detector = YoloDetector(model_name='/home/frinksserver/Desktop/Abhishek/object_tracking/best_biscuit_detection.pt')


    object_tracker = CentroidTracker()
    (H,W) = (None, None)

    result = cv2.VideoWriter('output1.mp4',  
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            10, size) 

    while cap.isOpened():

        success, img = cap.read()

        results = detector.score_frame(img)

        if img is None:
            return
        img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.3)
        coordinates= detections[0]

        coordinates = [tuple(coordinates)]

        start = time.perf_counter()
        tracks = object_tracker.update(coordinates)


        # loop over the tracked objects
        for (objectID, centroid) in tracks.items():
            print(objectID, centroid)
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        result.write(img)
        # show the output img
        cv2.imshow("img", img)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


    cap.release() 
    result.release()

    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
