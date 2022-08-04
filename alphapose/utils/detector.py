import os
import sys
from threading import Thread
from queue import Queue
import pdb
import cv2
import numpy as np

import torch
import torch.multiprocessing as mp

from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.models import builder

class DetectionLoader():
    def __init__(self, input_source, detector, cfg, opt, mode='image', batchSize=1, queueSize=128):
        self.cfg = cfg
        self.opt = opt
        self.mode = mode
        self.device = opt.device

        if mode == 'image':
            self.img_dir = opt.inputpath
            self.imglist = [os.path.join(self.img_dir, im_name.rstrip('\n').rstrip('\r')) for im_name in input_source]
            self.datalen = len(input_source)
        elif mode == 'video':
            stream = cv2.VideoCapture(input_source) # 將影片檔案讀入
            assert stream.isOpened(), 'Cannot capture source'
            self.path = input_source                                 # 影片路徑
            self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
            self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
            self.fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
            self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 影片長寬
            self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': self.frameSize} # 影片資訊
            stream.release()

        self.detector = detector       # 使用的 物件偵測模型
        self.batchSize = batchSize
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE        # 預設 姿態偵測模型的輸入維度
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE     # 預設 姿態偵測模型的輸出維度

        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == 'simple':
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN) # 讀取訓練資料集
            self.transformation = SimpleTransform(           # 客製化 pytorch transforms (在pytorch訓練前，對輸入資料做的影像處理)
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)
        elif cfg.DATA_PRESET.TYPE == 'simple_smpl':
            # TODO: new features
            from easydict import EasyDict as edict
            dummpy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': (2.2, 2.2, 2.2)
            })
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2.2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False,
                loss_type=cfg.LOSS['TYPE'])

        # initialize the queue used to store data
        """
        image_queue: the buffer storing pre-processed images for object detection
        det_queue: the buffer storing human detection results
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        if opt.sp:
            self._stopped = False
            self.image_queue = Queue(maxsize=queueSize)
            self.det_queue = Queue(maxsize=10 * queueSize)
            self.pose_queue = Queue(maxsize=10 * queueSize)
        else:
            self._stopped = mp.Value('b', False)
            self.image_queue = mp.Queue(maxsize=queueSize)
            self.det_queue = mp.Queue(maxsize=10 * queueSize)
            self.pose_queue = mp.Queue(maxsize=10 * queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        if self.mode == 'image':
            image_preprocess_worker = self.start_worker(self.image_preprocess)    # image 執行續0: image_preprocess
        elif self.mode == 'video':
            image_preprocess_worker = self.start_worker(self.frame_preprocess)    # video 執行續0: frame_preprocess(多次 image process)
        # start a thread to detect human in images
        image_detection_worker = self.start_worker(self.image_detection)          # 執行續1: 對影像進行 物件偵測
        # start a thread to post process cropped human image for pose estimation
        image_postprocess_worker = self.start_worker(self.image_postprocess)      # 執行續2:

        return [image_preprocess_worker, image_detection_worker, image_postprocess_worker]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        if self.opt.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.det_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def image_preprocess(self):
        for i in range(self.num_batches):
            imgs = []
            orig_imgs = []
            im_names = []
            im_dim_list = []
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                if self.stopped:
                    self.wait_and_put(self.image_queue, (None, None, None, None))
                    return
                im_name_k = self.imglist[k]

                # expected image shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(im_name_k)
                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)
                orig_img_k = cv2.cvtColor(cv2.imread(im_name_k), cv2.COLOR_BGR2RGB) # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
                im_dim_list_k = orig_img_k.shape[1], orig_img_k.shape[0]

                imgs.append(img_k)
                orig_imgs.append(orig_img_k)
                im_names.append(os.path.basename(im_name_k))
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                imgs = torch.cat(imgs)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                # im_dim_list_ = im_dim_list

            self.wait_and_put(self.image_queue, (imgs, orig_imgs, im_names, im_dim_list))

    def frame_preprocess(self):
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), 'Cannot capture source'

        for i in range(self.num_batches):
            imgs = []
            orig_imgs = []
            im_names = []
            im_dim_list = []
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                (grabbed, frame) = stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed or self.stopped:
                    # put the rest pre-processed data to the queue
                    if len(imgs) > 0:
                        with torch.no_grad():
                            # Record original image resolution
                            imgs = torch.cat(imgs)
                            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                        self.wait_and_put(self.image_queue, (imgs, orig_imgs, im_names, im_dim_list))
                    self.wait_and_put(self.image_queue, (None, None, None, None))
                    print('===========================> This video get ' + str(k) + ' frames in total.')
                    sys.stdout.flush()
                    stream.release()
                    return

                # expected frame shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(frame)

                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)

                im_dim_list_k = frame.shape[1], frame.shape[0]

                imgs.append(img_k)
                orig_imgs.append(frame[:, :, ::-1])
                im_names.append(str(k) + '.jpg')
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Record original image resolution
                imgs = torch.cat(imgs)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                # im_dim_list_ = im_dim_list

            self.wait_and_put(self.image_queue, (imgs, orig_imgs, im_names, im_dim_list))
        stream.release()

    def image_detection(self):   # 物件偵測
        print('detector')
        #pdb.set_trace() # use it when debug mode
        for i in range(self.num_batches):
            imgs, orig_imgs, im_names, im_dim_list = self.wait_and_get(self.image_queue)  # 從frames 當中一次提取  num_batches個frames
            # 假設 self.num_batches = 5 , imgs = [5,3,608,608] 5張rgb的圖片
            # orig_imgs: 5個Numpy Array； im_names = ['0.jpg','1.jpg','2.jpg','3.jpg','4.jpg'] (從image_queue提取連續的5張frame的名稱)
            #　im_dim_list：　圖片的長與寬(維度)
            if imgs is None or self.stopped:
                self.wait_and_put(self.det_queue, (None, None, None, None, None, None, None)) 
                return

            with torch.no_grad():
                # pad useless images to fill a batch, else there will be a bug
                for pad_i in range(self.batchSize - len(imgs)):
                    imgs = torch.cat((imgs, torch.unsqueeze(imgs[0], dim=0)), 0)
                    im_dim_list = torch.cat((im_dim_list, torch.unsqueeze(im_dim_list[0], dim=0)), 0)

                dets = self.detector.images_detection(imgs, im_dim_list) # 開始偵測。!!!!!!!!! (detectoe\api.py : yolo_api.py)
                # dets: 偵測出的結果是座標、還有偵測的準確度。
                # dets : 紀載了5張圖片的boundingbox的座標、與評分。  dets的第一航表示是第幾(1~5張)(6~10張)...。
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_imgs)):
                        self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], None, None, None, None, None)) # 儲存 物件偵測結果
                    continue
                if isinstance(dets, np.ndarray):
                    dets = torch.from_numpy(dets)
                dets = dets.cpu()
                boxes = dets[:, 1:5] # 所有候選框的 角落座標
                scores = dets[:, 5:6] # 所有候選框的 評分
                if self.opt.tracking:
                    ids = dets[:, 6:7]
                else:
                    ids = torch.zeros(scores.shape)

            for k in range(len(orig_imgs)):
                boxes_k = boxes[dets[:, 0] == k] #　dets[:, 0]表達的是第 k 張圖片。 boxes_k: 第k個frame的偵測出的bounding_box
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], None, None, None, None, None))
                    continue
                inps = torch.zeros(boxes_k.size(0), 3, *self._input_size)
                cropped_boxes = torch.zeros(boxes_k.size(0), 4)
                # 將辨識完的結果丟到 Queue 裡面
                self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes))
                
                
    def image_postprocess(self):
        pdb.set_trace()  # use it when debug mode
        for i in range(self.datalen):
            with torch.no_grad():
                (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.wait_and_get(self.det_queue) # 將每個frame物件偵測的結果從 det_queue 取出。
                if orig_img is None or self.stopped:        # frame 抽完結束
                    self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
                    return
                if boxes is None or boxes.nelement() == 0:  # 沒有成功偵測出框的frame，直接丟入pose_queue(cropped_boxes=None)
                    self.wait_and_put(self.pose_queue, (None, orig_img, im_name, boxes, scores, ids, None))
                    continue
                # imght = orig_img.shape[0]
                # imgwidth = orig_img.shape[1]
                for i, box in enumerate(boxes):             # 有成功框出物件的(物件偵測成功的frame) 丟入 pose_queue
                    # boxes:yolo(或其他)輸出的bounding_box是在原圖上的座標(不只一個)。 但之後我們要丟去做pose estimation，所以我們要reshape。
                    inps[i], cropped_box = self.transformation.test_transform(orig_img, box) # 將"原圖"與"bounding_box"丟進去做影像處理，框出的物件reshape成256*192的圖。
                    # inps 就是被框出的物件 另存一張圖。
                    cropped_boxes[i] = torch.FloatTensor(cropped_box)

                # inps, cropped_boxes = self.transformation.align_transform(orig_img, boxes)

                self.wait_and_put(self.pose_queue, (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)) 

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def length(self):
        return self.datalen
