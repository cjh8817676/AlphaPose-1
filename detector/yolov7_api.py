# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn)
# -----------------------------------------------------

"""API of YOLOv7 detector"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np

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
        self.model_weights = cfg.get("MODEL_WEIGHTS", "detector/yolov7/data/yolov7.pth")
        self.num_classes = self.exp.num_classes
        self.conf_thres = cfg.get("CONF_THRES", 0.1)
        self.nms_thres = cfg.get("NMS_THRES", 0.6)
        self.inp_dim = cfg.get("INP_DIM", 640)
        self.img_size = [self.inp_dim, self.inp_dim]

        self.model = None

    def load_model(self):
        args = self.detector_opt
        # Load model
        print(f"Loading {self.model_name.upper().replace('_', '-')} model..")
        device =  select_device(self.opt.device)   # 使用gpu
        
        self.model = self.exp.get_model()
        self.model.load_state_dict(
            torch.load(self.model_weights, map_location="cpu")["model"]
        )

        if args:
            if len(args.gpus) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus).to(
                    args.device
                )
            else:
                self.model.to(args.device)
        else:
            self.model.cuda()
        self.model.eval()

    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)
        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))
        """
        # if isinstance(img_source, str):
        #     img, orig_img, im_dim_list = prep_image(img_source, self.img_size)
        # elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):
        #     img, orig_img, im_dim_list = prep_frame(img_source, self.img_size)
        # else:
        #     raise IOError("Unknown image source type: {}".format(type(img_source)))

        # return img
        pass

    def images_detection(self, imgs, orig_dim_list):
        """
        Feed the img data into object detection network and
        collect bbox w.r.t original image size
        Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input
               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size
        Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results
        """


    def dynamic_write_results(
        self, prediction, num_classes, conf_thres, nms_thres, classes=0
    ):
        pass

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
