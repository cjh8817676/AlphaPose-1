from easydict import EasyDict as edict

cfg = edict()
cfg.MODEL_NAME = "yolov7"
cfg.MODEL_WEIGHTS = "detector/yolov7/data/yolov7.pt"
cfg.INP_DIM = 640
cfg.CONF_THRES = 0.1
cfg.NMS_THRES = 0.6
