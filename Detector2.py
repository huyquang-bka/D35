# Common
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Torch
import torch
import torch.backends.cudnn as cudnn

# QT
from PyQt5 import QtCore

# Misc
from models.common import DetectMultiBackend, letterbox
from utils.general import (check_img_size, cv2, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
import time
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import opt
from utils.general import xyxy2xywh
import requests
import cv2
import base64
import json
from sort import *


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# MAIN
class DetectorThread(QtCore.QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index=0, parent=None):
        super().__init__()
        # super(DetectorThread, self).__init__(parent)
        # id
        self.is_running = True
        self.index = index
        # VARIABLES
        self.source = r"C:\Users\Admin\Downloads\video-1636524259.mp4"
        self.weights = 'yolov5s.pt'
        self.data = 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz = 1280  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = 0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.save_crop = False  # save cropped prediction boxes
        # filter by class: --class 0, or --class 0 2 3
        self.classes = [2, 5]  # (80) range of classes
        self.agnostic_nms = True  # class-agnostic NMS
        self.line_thickness = 1  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = True  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def setup(self, a_source):
        self.source = a_source

    @torch.no_grad()
    def run(self):
        # Load deepsort
        sort_tracker = Sort(max_age=25 * 5,
                            min_hits=2,
                            iou_threshold=0.5)
        # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.model.half() if self.half else model.model.float()
        print('Starting thread...', self.index)
        self.source = str(self.source)
        cap = cv2.VideoCapture(self.source)
        count = 0
        while self.is_running:
            s = time.time()
            cap.grab()
            ret, img0 = cap.retrieve()
            img0 = cv2.resize(img0, (1280, 720))
            if not ret:
                count += 1
                if count == 5:
                    break
                time.sleep(3)
                continue
            count += 1
            im0 = img0.copy()
            img = letterbox(img0, self.imgsz, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(img)
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
            dets_to_sort = np.empty((0, 6))

            # Pass detections to SORT
            # NOTE: We send in detected object class too
            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

            # Run SORT
            tracked_dets = sort_tracker.update(dets_to_sort)

            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                for bbox, detclass, identity in zip(bbox_xyxy, categories, identities):
                    x1, y1, x2, y2 = list(map(int, bbox))
                    identity = int(identity)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), compute_color_for_id(detclass), 2)
                    cv2.putText(im0, str(identity), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                compute_color_for_id(detclass),
                                2)

            FPS = 1 // (time.time() - s)
            cv2.putText(im0, '%g FPS' % FPS, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.signal.emit(im0)
            print("Time emit:", time.time() - s)
            print(f"Thread {self.index} real FPS: {1 // (time.time() - s)}")

            cv2.waitKey(1)

    def stop(self):
        print('Stopping thread...', self.index)
        self.is_running = False


class DetectorFree(QtCore.QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index=0, parent=None):
        super().__init__()
        # super(DetectorThread, self).__init__(parent)
        # id
        self.is_running = True
        self.index = index
        # VARIABLES
        self.source = 0

    def setup(self, a_source):
        self.source = a_source

    @torch.no_grad()
    def run(self):
        # Load deepsort
        print('Starting thread...', self.index)
        self.source = 0 if self.source in [0, "0"] else str(self.source)
        print(self.source, type(self.source))
        cap = cv2.VideoCapture(self.source)
        while self.is_running:
            s = time.time()
            cap.grab()
            ret, image = cap.retrieve()
            if not ret:
                break
            # print(image.shape)
            # FPS = 1 // (time.time() - s)
            # cv2.putText(image, '%g FPS' % FPS, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # print(f"Thread {self.index} FPS: {FPS}")
            self.signal.emit(image)
            cv2.waitKey(1)

    def stop(self):
        print('Stopping thread...', self.index)
        self.is_running = False
