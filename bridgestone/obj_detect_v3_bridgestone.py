from scipy.spatial import distance as dist
import cv2
import torch
import numpy as np


class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]



    def update(self, rects):
        new_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (start_x, start_y, end_x, end_y,_)) in enumerate(rects):
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            new_centroids[i] = (c_x, c_y)

        if len(self.objects) == 0:
            for i in range(len(new_centroids)):
                self.register(new_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            previous_centroids = np.array(list(self.objects.values()))

            D = dist.cdist(previous_centroids, new_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Keep track of used rows and columns
            used_rows = set()
            used_cols = set()

            if D.size > 0:
                rows = D.min(axis=1).argsort()
                
                cols = D.argmin(axis=1)[rows]

                # Update tracking info
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue

                    object_id = object_ids[row]
                    self.objects[object_id] = new_centroids[col]

                    used_rows.add(row)
                    used_cols.add(col)

                # Register new centroids
                for col in set(range(0, D.shape[1])).difference(used_cols):
                    self.register(new_centroids[col])
                    
                # Deregister old centroids
                for row in set(range(0, D.shape[0])).difference(used_rows):
                    object_id = object_ids[row]
                    self.deregister(object_id)
            else:
                print('skipping D is empty')

                # Register new centroids
                for col in set(range(0, D.shape[1])).difference(used_cols):
                    self.register(new_centroids[col])
                # Deregister old centroids
                for row in set(range(0, D.shape[0])).difference(used_rows):
                    object_id = object_ids[row]
                    print(object_id)
                    self.deregister(object_id)




def main():
    # Initialize YOLOv5 model
    model_name = '/home/frinksserver/Desktop/Abhishek/object_tracking/centroid_tracker/bridgestone/best.pt'
    model = torch.hub.load('/home/frinksserver/Desktop/Abhishek/object_tracking/yolov5', "custom", source='local',path=model_name,force_reload=True)

    # Initialize centroid tracker
    tracker = CentroidTracker()

    # Initialize Video Capture
    cap = cv2.VideoCapture('/home/frinksserver/Desktop/Abhishek/object_tracking/centroid_tracker/bridgestone/newBridgestoneTest.mp4')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)
    fps = int(cap.get(5))



    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    size = (frame_width,frame_height)
    print(size)


    start_point = (frame_width//2,0)
    end_point = (frame_width//2,frame_height)

    result = cv2.VideoWriter('/home/frinksserver/Desktop/Abhishek/object_tracking/centroid_tracker/bridgestone/bridge_output.mp4',  
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, size) 

    count = 0
    obj_lt = []
    obj_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Run YOLOv5 model on the frame
        results = model(frame)
        _, cord = results.xyxy[0][:, -1].cpu().numpy(), results.xyxy[0][:, :-1].cpu().numpy()
        
        x = []

        boxes = cord
        

        for (i, (start_x, start_y, end_x, end_y, _)) in enumerate(boxes):
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)



        # Update left tracker
        tracker.update(boxes)

    
        # Draw tracked objects
        for (object_id, centroid) in tracker.objects.items():
            
            cv2.circle(frame, (centroid[0], centroid[1]), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {object_id}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2) 
            x.append(object_id)
            if centroid[0]>(frame_width//2) and object_id not in obj_lt:
                obj_count+=1
                obj_lt.append(object_id)
            

        cv2.putText(frame, f"ID: {x}", (1500,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        cv2.putText(frame, f"object count: {obj_count}", (1500,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
        cv2.imwrite(f'/home/frinksserver/Desktop/Abhishek/object_tracking/centroid_tracker/bridgestone/output/{count}.jpg',frame)
        result.write(frame)
        count+=1
        cv2.imshow('Frame', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
