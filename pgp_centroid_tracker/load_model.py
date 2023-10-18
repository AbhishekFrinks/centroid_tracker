import os, sys, cv2, torch, datetime
import numpy as np
from ultralytics import YOLO


class YoloV8:
    def __init__(self, model_weights, classes, score_thresh=0.7, iou_thres=0.3, device="cuda", imgsz=(640, 640)):
        self.model_weights = model_weights # Model weights
        self.classes = classes # Classes the model have to detect
        self.imgsz = imgsz # Image size model expects
        self.score_thresh = score_thresh # Confidence threshold
        # self.iou_thresh = iou_thres # IOU threshold
        self.device = device # device to run the inference on 
        self.model = self.load_model() # Loading the yolo model

    # Method to load the yolov8 model
    def load_model(self):
        return YOLO(self.model_weights) 

    # Method to preprocess the image -- *nothing to do here
    def preProcess(self, image):
        return image

    # Method to pass image through the model
    def forward(self, image):
        pred = self.model.predict(device=self.device, source=image, imgsz=self.imgsz, conf=self.score_thresh,  verbose=False)
        return pred[0]

    # Method to postprocess the result in proper format
    def postProcess(self, pred_output):
        pred_boxes_array = [list(np.array(i).astype(int)) for i in pred_output.boxes.xyxy.cpu()]
        scores = [i for i in pred_output.boxes.conf.cpu()]
        classes = [int(i) for i in pred_output.boxes.cls.cpu()]
        return pred_boxes_array, classes, scores

    # Method to put the model in specified device
    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)

    # Method to call for direct inferencing
    def __call__(self, image):
        input = self.preProcess(image=image)
        pred_output = self.forward(image=input)
        result = self.postProcess(pred_output=pred_output)
        boxes, classes, scores = result
        return boxes, classes, scores