# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Chao Xu (xuchao.19962007@sjtu.edu.cn)
# -----------------------------------------------------

"""API of detector"""
from abc import ABC, abstractmethod

# 選擇使用哪一種 detector
def get_detector(opt=None):
    if opt.detector == 'yolo':                       # default detector
        from detector.yolo_api import YOLODetector
        from detector.yolo_cfg import cfg
        return YOLODetector(cfg, opt)
    elif 'yolox' in opt.detector:
        from detector.yolox_api import YOLOXDetector
        from detector.yolox_cfg import cfg
        if opt.detector.lower() == 'yolox':
            opt.detector = 'yolox-x'
        cfg.MODEL_NAME = opt.detector.lower()
        cfg.MODEL_WEIGHTS = f'detector/yolox/data/{opt.detector.lower().replace("-", "_")}.pth'
        return YOLOXDetector(cfg, opt)
    elif 'yolov7' in opt.detector:
        from detector.yolov7_api import YOLOV7Detector
        from detector.yolov7_cfg import cfg
        if opt.detector.lower() == 'yolov7':
            opt.detector = 'yolov7'
        cfg.MODEL_NAME = opt.detector.lower()
        cfg.MODEL_WEIGHTS = f'detector/yolov7/{opt.detector.lower().replace("-", "_")}.pt'  # 預訓練模型存放位置
        return YOLOV7Detector(cfg, opt)
    elif opt.detector == 'tracker':
        from detector.tracker_api import Tracker
        from detector.tracker_cfg import cfg
        return Tracker(cfg, opt)
    elif opt.detector.startswith('efficientdet_d'):
        from detector.effdet_api import EffDetDetector
        from detector.effdet_cfg import cfg
        return EffDetDetector(cfg, opt)
    else:
        raise NotImplementedError


class BaseDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def image_preprocess(self, img_name):
        pass

    @abstractmethod
    def images_detection(self, imgs, orig_dim_list):
        pass

    @abstractmethod
    def detect_one_img(self, img_name):
        pass
