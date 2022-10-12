# -*- coding: utf-8 -*-

"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm
import natsort

from PIL import Image
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track

from detector.apis import get_detector

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter
from alphapose.utils.writer import DataWriter

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

def check_input():
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # for detection results
    if len(args.detfile):
        if os.path.isfile(args.detfile):
            detfile = args.detfile
            return 'detfile', detfile
        else:
            raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]

        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 影像侵蝕 (影像侵蝕對於移除影像中的小白雜點很有幫助，可用來去噪，例如影像中的小雜點，雜訊。)
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # 影像膨脹

if __name__ == "__main__":
    mode, input_source = check_input()   # model: image、webcame、video
    
    # load yolo 
    yolo_detector = get_detector(args)
    yolo_detector.load_model()
    
    print(os.getcwd())
    
    from depth_estimation.AdaBins.models import UnetAdaptiveBins
    from depth_estimation.AdaBins import model_io
    from depth_estimation.AdaBins.infer import InferenceHelper
    
    #load depth_estimation model
    inferHelper = InferenceHelper(dataset='nyu')    
    
# ==========================  video info  =====================================

    stream = cv2.VideoCapture(input_source)
    path = input_source                                 # 影片路徑
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
    fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) # 影片寬
    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 影片長
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': (h,w)} # 影片資訊
    orig_dim_list = torch.Tensor([[w,h,w,h]])
    
# =============================================================================
            
    
    while stream.isOpened():
        ret, frame = stream.read()                                # frame : (origin_w,origin_h,3)的 Array
       
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame= cv2.resize(frame,(640,480), interpolation = cv2.INTER_AREA) 
        
        # yolo detect people
        preprocess_frame = yolo_detector.image_preprocess(frame)  # 將原圖resize 成(1,3,model_weight,model_height) 的Tensor
        dets = yolo_detector.images_detection(preprocess_frame, orig_dim_list)
        
        # depth estimation and some processing
        centers, pred = inferHelper.predict_pil(frame)
        pred = pred.squeeze()
        pred = np.reshape(pred,(pred.shape[0],pred.shape[1],1))
    
        grey_3_channel = cv2.cvtColor(pred.squeeze(), cv2.COLOR_GRAY2RGB)  # 將 1-D 深度 灰階圖 轉成 3D 灰階圖
        grey_3_channel = cv2.normalize(grey_3_channel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        
        if isinstance(dets, int) or dets.shape[0] == 0: # 沒有成功偵測到人就繼續
            print('int',type(dets))
            numpy_horizontal_concat = np.concatenate((frame, grey_3_channel ), axis=1)
            cv2.imshow('detection', numpy_horizontal_concat)     
        else:
            if isinstance(dets, np.ndarray):                # 有成功偵測到人
                print('ndarray',type(dets))
                dets = torch.from_numpy(dets)
                
            dets = dets.cpu()
            boxes = dets[:, 1:5] # 所有候選框的 角落座標
            scores = dets[:, 5:6] # 所有候選框的 評分    
            
            if args.tracking:
                ids = dets[:, 6:7]
            else:
                ids = torch.zeros(scores.shape)
                
            boxes_k = boxes[dets[:, 0] == 0]
            
            # render image  (沒偵測到人，用演算法)
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                print('missing boundingbox',frame)
                continue
    
            for i in range(boxes_k.shape[0]):
                if scores[i] > 0.4:  # yolo 的置信度
                    cv2.rectangle( frame , (int(boxes_k[i][0]), int(boxes_k[i][1])), (int(boxes_k[i][2]), int(boxes_k[i][3])), (0, 0, 255), 3)
            
         
        numpy_horizontal_concat = np.concatenate((frame, grey_3_channel ), axis=1)
        
        cv2.imshow('detection', numpy_horizontal_concat)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    
    stream.release()
    cv2.destroyAllWindows()
        
        
        
        
        
        
        
    
    

    
                                        
