
import cv2
import numpy as np
import sys
import glob
import time
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from deep_sort_realtime.deepsort_tracker import DeepSort


class YoloDetector():

    def __init__(self, model_name):
        
        self.model = self.load_model(model_name)
        self.classes = self.model.names

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print("Using Device: ", self.device)
        

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
                
                x_center = x1 + (x2 - x1)
                y_center = y1 + ((y2 - y1) / 2)
                
                tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                confidence = float(row[4].item())
                feature = self.class_to_label(labels[i])

                detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), feature))
                
        
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


    object_tracker = DeepSort(max_age=5,
                    n_init=2,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.3,
                    nn_budget=None,
                    override_track_class=None,
                    embedder="mobilenet",
                    half=True,
                    bgr=True,
                    embedder_gpu=True,
                    embedder_model_name=None,
                    embedder_wts=None,
                    polygon=False,
                    today=None)

    result = cv2.VideoWriter('output1.mp4',  
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            10, size) 

    while cap.isOpened():

        succes, img = cap.read()
    
        
        results = detector.score_frame(img)

        if img is None:
            return
        img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.3)

        start = time.perf_counter()
        tracks = object_tracker.update_tracks(detections, frame=img) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

        end = time.perf_counter()
        totalTime = end - start
        print(totalTime)
        for track in tracks:
            if not track.is_confirmed():
                pass
            track_id = track.track_id

            ltrb = track.to_ltrb()
            
            bbox = ltrb
            
            cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
            cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        result.write(img)
            
        fps = 1 / totalTime


        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('img',img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release() 
    result.release()

    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
