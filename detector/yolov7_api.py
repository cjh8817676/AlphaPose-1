# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn)
# -----------------------------------------------------

"""API of YOLOv7 detector"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'yolov7'))
import torch
import numpy as np
import pdb
from detector.apis import BaseDetector
import time
from pathlib import Path
import cv2
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class YOLOV7Detector(BaseDetector):
    def __init__(self, cfg, opt=None):
        super(YOLOV7Detector, self).__init__()

        self.detector_cfg = cfg
        
        self.detector_opt = opt
        self.model_name = cfg.get("MODEL_NAME", "yolov7")
        self.model_weights = cfg.get("MODEL_WEIGHTS", "detector/yolov7/data/yolov7.pt")
        self.num_classes = 80 # 先固定
        self.conf_thres = cfg.get("CONF_THRES", 0.1)
        self.nms_thres = cfg.get("NMS_THRES", 0.6)
        self.inp_dim = cfg.get("INP_DIM", 640)
        self.img_size = self.inp_dim
        self.stride = 32   # 先固定
        self.model = None

    def load_model(self):
        args = self.detector_opt
        
        self.model = attempt_load(self.model_weights,args.device)
        print(self.model)
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.inp_dim, s=stride)  # check img_size
        
        self.model = self.model.half()
        self.model(torch.zeros(1, 3, imgsz, imgsz).to(args.device).type_as(next(self.model.parameters())))  # run once
        

    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)
        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))
        """
        # pdb.set_trace() 
        img = self.letterbox(img_source, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        return img

    def images_detection(self, imgs, orig_dim_list):
        """
        Feed the img data into object detection network and
        collect bbox w.r.t original image size
        Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input
               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size
        Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results
        c:  物件辨識準確度
        s:  物件追蹤準確度
        idx of cls:  trace id (給bounding box 一個id)
        """
        args = self.detector_opt
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        if not self.model:
            self.load_model()   # 模型載入
         
        device = select_device('')   # 使用gpu
        half = device.type != 'cpu'  # half precision only supported on CUDA

        with torch.no_grad():
            imgs = imgs.to(args.device) if args else imgs.cuda()
            img = imgs.half() if half else imgs.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            
            prediction = self.model(img,augment = False)[0]                    # 模型預測
            # do nms to the detection results, only human category is left
            dets = self.dynamic_write_results(
                prediction,
                num_classes=self.num_classes,
                conf_thres=self.conf_thres,
                nms_thres=self.nms_thres,
                classes=0,
            )
            if isinstance(dets, int) or dets.shape[0] == 0:
                return 0
            # 這裡的 dets 的候選框座標 是從模型輸出的座標系框出來的。 我們要將它映射回原圖的座標系。
            # pdb.set_trace()
            im0_shape = (int(orig_dim_list[0][1]),int(orig_dim_list[0][0]),3) # (h,w,3)
            dets[:, 1:5] = scale_coords(img.shape[2:], dets[:, 1:5], im0_shape).round() # 將候選框映射回原圖
            
            dets = dets.cpu()
            
            return dets
        
        

    def dynamic_write_results(
        self, prediction, num_classes, conf_thres, nms_thres, classes=0
    ):
        prediction_bak = prediction.clone()
        # pdb.set_trace()
        dets = non_max_suppression (
            prediction.clone(),
            conf_thres=conf_thres,
            iou_thres=0.45,
            classes=0,
            agnostic = False
        )
        # pdb.set_trace()
        if isinstance(dets, int):
            return dets
        

        return dets

    def detect_one_img(self, img_name):
        """
        Detect bboxs in one image
        Input: 'str', full path of image
        Output: '[{"category_id":1,"score":float,"bbox":[x,y,w,h],"image_id":str},...]',
        The output results are similar with coco results type, except that image_id uses full path str
        instead of coco %012d id for generalization.
        """
        args = self.detector_opt
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        if not self.model:
            self.load_model()
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        pass
    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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