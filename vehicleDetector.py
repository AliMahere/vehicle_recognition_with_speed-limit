from statistics import mode
import cv2
from sympy import true
import torch
import time
import numpy as np
import argparse 
import random
from cv2 import imshow
from utilities import set_logging , select_device , attempt_load , time_synchronized, non_max_suppression, scale_coords, xyxy2xywh

class Detector():

    def __init__(self, modelPath = "weights/best.pt" ):
        """FaceDetctor class used to detect faces insde a frame .
        Args:
            device (int, optional): [description]. Defaults to 0.
        """
        set_logging()
        self.device = select_device('cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = self.setModeltest(modelPath)
        self.stride = int(self.model.stride) 
        self.imgsz = 416
        self.conf = 0.4
        self.iou_thres = 0.45 
 




    def letterbox(self,img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def setModel(self,path, half = False):
        model = attempt_load(path, map_location=self.device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride

    def setModeltest(self,path, half = False):
        
        model = torch.hub.load('ultralytics/yolov5', 'custom', path = path)
#        model = model.autoshape()  # for PIL/cv2/np inputs and NMS
        return model
    def setModeltest2(self,path, half = False):
        
        model = attempt_load(path, map_location=self.device) # load FP32 model 
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
        #imgsz = check_img_size(imgsz, s=model.stride.max()) # check img_size if half: model.half() # to FP16
        return model

    def getImage(self, img0):
#        img0 = cv2.imread(path)  # BGR
#        assert img0 is not None, 'Image Not Found ' + path
#        print(f'image {self.count}/{self.nf} {path}: ', end='')
        img = self.letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0


    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, (255, 255, 0), thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



    def detect(self, img):
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        
        t0 = time.time()


        img, img0 = self.getImage(img)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img)
        pred = non_max_suppression(pred, 0.25, 0.45)
        t2 = time_synchronized()


        # Process detections
        bboxesCid = []
        for i, det in enumerate(pred):  # detections per image
            
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    label = f'{names[int(cls)]} {conf:.2f}'
                    self.plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])
                    bboxesCid.append([x1, y1, x2, y2, int(cls)])

              #      if save_img or view_img:  # Add bbox to image
             #           label = f'{names[int(cls)]} {conf:.2f}'
              #          plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Stream results

        print(f'Done. ({time.time() - t0:.3f}s)')

        return img0, bboxesCid





# d = Detector()
# img = "images/00e481ea1a520175_jpg.rf.MV6sZ8QCFwFeMYaI2tHm.jpg"
# out , data = d.detect(img)
# print(data)
# imshow("out", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
