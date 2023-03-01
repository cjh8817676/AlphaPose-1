"""
Based on https://github.com/seoungwugoh/ivs-demo

The entry point for the user interface
It is terribly long... GUI code is hard to write!
"""
import sys
import argparse
import platform
import os
from os import path
import functools
from argparse import ArgumentParser
from queue import Queue
import time
import cv2
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import natsort
from collections import deque
import pdb
from PyQt5.QtWidgets import (QWidget, QApplication, QComboBox, 
    QHBoxLayout, QLabel, QPushButton, QTextEdit, 
    QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, 
    QShortcut, QRadioButton, QProgressBar, QFileDialog)

from PyQt5.QtGui import QPixmap, QKeySequence, QImage, QTextCursor
from PyQt5.QtCore import Qt, QTimer 
from PyQt5 import QtCore

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter


# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '' 
file_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_path,'../XMem-1'))
# sys.path.insert(0, 'C:\mydesktop\Gymnastic_Plan\workspace\XMem-1')
from model.network import XMem
from inference.interact.s2m_controller import S2MController
from inference.interact.fbrs_controller import FBRSController
from inference.interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M

from PyQt5.QtWidgets import QApplication
from inference.interact.gui import App
from inference.interact.resource_manager import ResourceManager

torch.set_grad_enabled(False)


"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Mivos Demo')
parser.add_argument('--missing_bbox', default=False, required=False,
                    help='Only get largest object')
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
parser.add_argument('--video', help='Video file readable by OpenCV. Either this or --images needs to be specified.', 
                    default= os.path.join(file_path,'../test_video/cat_jump.mp4'))
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

"""----------------------------- MiVOS options -----------------------------"""
parser.add_argument('--model', default=os.path.join(file_path,'../XMem-1/saves/XMem.pth'))
parser.add_argument('--s2m_model', default=os.path.join(file_path,'../XMem-1/saves/s2m.pth'))
parser.add_argument('--fbrs_model', default=os.path.join(file_path,'../XMem-1/saves/fbrs.pth'))

"""
Priority 1: If a "images" folder exists in the workspace, we will read from that directory
Priority 2: If --images is specified, we will copy/resize those images to the workspace
Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
That way, you can continue annotation from an interrupted run as long as the same workspace is used.
"""
parser.add_argument('--images', help='Folders containing input images.', default=None)
# parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
parser.add_argument('--workspace', help='directory for storing buffered images (if needed) and output masks', default=None)

parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=100)

parser.add_argument('--num_objects', type=int, default=1)

# Long-memory options
# Defaults. Some can be changed in the GUI.
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128) 

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', type=int, default=10)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
parser.add_argument('--size', default=480, type=int, 
        help='Resize the shorter side to this size. -1 to use original resolution.(or you cant set --size as same as width of video) ')
args = parser.parse_args()

torch.set_grad_enabled(False)
cfg = update_config(args.cfg)   # cfg of pose model
config = vars(args)
config['enable_long_term'] = True
config['enable_long_term_count_usage'] = True

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


def pose_estimate(args,cfg,mode,det_loader,det_worker):
    # Load pose model   
    print('Load Pose Model')
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if args.pose_track:                 #Tracker 的使用
        tracker = Tracker(tcfg, args)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }
    # pdb.set_trace()
    # Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize
    if args.save_video and mode != 'image':
        from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
        if mode == 'video':
            pose_net_name = args.cfg.split('/')[-1]
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' +args.detector +'_'+ pose_net_name.split('.')[0]  + os.path.basename(input_source))
        else:
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()
        # writer: 
        
    else:
        writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start() # 開始渲染圖片， (可決定是否存檔)

    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read() # 讀取物件偵測完成的frames，並做pose estimation 
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j) # 在做pose estimation的時候，使用的是tensor。(hm_j : 人數,keypoint數,長,寬) hm:heat_map
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                if args.pose_track:
                    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)# 給予bounding box編號
                hm = hm.cpu()  # hm: heat_map 
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
                
                
        print_finish_info()
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering0 remaining ' + str(writer.count()) + ' images in the queue...')
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()
        det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering1 remaining ' + str(writer.count()) + ' images in the queue...')

                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()

if __name__ == '__main__':
    mode, input_source = check_input()   # model: image、webcame、video
    
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    
    if args.detector == 'mivos':
        with torch.cuda.amp.autocast(enabled=not args.no_amp):

            # Load our checkpoint
            network = XMem(config, args.model).cuda().eval()

            # Loads the S2M model
            if args.s2m_model is not None:
                s2m_saved = torch.load(args.s2m_model)
                s2m_model = S2M().cuda().eval()
                s2m_model.load_state_dict(s2m_saved)
            else:
                s2m_model = None

            s2m_controller = S2MController(s2m_model, args.num_objects, ignore_class=255)  # 滑鼠點物件，形成mask功能 的物件
            if args.fbrs_model is not None:
                fbrs_controller = FBRSController(args.fbrs_model)
            else:
                fbrs_controller = None

            # Manages most IO
            resource_manager = ResourceManager(config)

            app = QApplication(sys.argv)
            ex = App(network, resource_manager, s2m_controller, fbrs_controller,config, args, cfg)
            sys.exit(app.exec_())
    else:
        # Manages most IO
        resource_manager = ResourceManager(config)
        # Load detection loader
        if mode == 'webcam':
            det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
            det_worker = det_loader.start()
        elif mode == 'detfile':
            det_loader = FileDetectionLoader(input_source, cfg, args)
            det_worker = det_loader.start()
        else:
            det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
            det_worker = det_loader.start()

    pose_estimate(args,cfg,mode,det_loader,det_worker)

        
    


