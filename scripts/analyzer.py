import sys,os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaMetaData
from PyQt5.QtMultimediaWidgets import QVideoWidget
import json
from PyQt5 import uic
import pandas as pd
import pdb
import glob
from PIL import Image
file_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_path,'../../XMem-1'))
from inference.interact.interactive_utils import *
from numba import jit
from collections import defaultdict
from defaultlist import defaultlist

imu_data_pd = pd.DataFrame()
imu_data_len = 0
imu_sample_rate = 400
imu_data_gyrox = []
imu_data_gyroy = []
imu_data_gyroz = []
imu_data_accx = []
imu_data_accy = []
imu_data_accz = []
imu_data_haccx = []
imu_data_haccy = []
imu_data_haccz = []
imu_data_left = 0
imu_data_right = 1200

# coco pose
'''
[nose,leye,reye,lear,rear,lshoulder,rshoulder,lelbow,relbow,lwrist,rwrist,lhip,rhip,lknee,rknee,lankle,rankle]
'''
# mask of human is red
@jit(nopython=True)
def find_human_mask(original_mask): # shape (w,h,3)  RGB
    # pdb.set_trace()
    h, w, c = original_mask.shape
    # human_mask = np.zeros((w,h,3)) # plt.imshow(human_mask) to verify if we only get human mask
    human_mask_point = []
    
    # for i in range(0,h):    # row
    #     for j in range(0,w):  # col
    #         if np.array_equal(original_mask[i][j],np.array([128,0,0])):
    #             # human_mask[i][j] = np.array([128,0,0])
    #             human_mask_point.append([j,i])  # [col, row] 為了之後方便計算,倒過來放
    
    rows,cols = np.where((original_mask[:,:,0]==128) & (original_mask[:,:,1]==0) & (original_mask[:,:,2]==0))
    
    for i,j in zip(rows,cols):
        human_mask_point.append([j,i])
        # print(i,j)
    
                
    return human_mask_point

# mask of high bar is green
@jit(nopython=True)
def find_horizontal_bar_mask(original_mask):
    # pdb.set_trace()
    w , h, c = original_mask.shape
    min_row = 4000
    min_col = 4000
    max_col = -1
    max_row = -1
    green = 0
    for i in range(0,h):     # row
        for j in range(0,w):
            if np.array_equal(original_mask[i][j],np.array([0,128,0])):
                min_row = min(min_row,i)
                max_row = max(max_row,i)
                min_col = min(min_col,j)
                max_col = max(max_col,j)
                green += 1
    if green == 0:
        print("no bar mask!")
                
    top_of_bar = [min_col , min_row]
    bottom_of_bar = [max_col , max_row]
                
    return [top_of_bar, bottom_of_bar]

# get closet distance from a sets and a point
@jit(nopython=True)
def get_mini_distance(point,mask_set):
    latest_distance =4000
    for i in mask_set:
        distance = np.sqrt(np.sum(np.square(point-i)))
        if latest_distance > distance:
            latest_distance = distance
    
    return latest_distance
# @jit(nopython=True)
def check_handoff(bar_point,human_point,human_mask,counter):
    # bar_point: left_top and roght bottom [[min_col,min_row],[max_col,max_row]] (row: h col : w)
    # human_mask: many points as mask, enery points is [col, row]
    # if counter == 25:
    #     pdb.set_trace()
    human_mask = human_mask[0]

    bar_highest_point = np.array([ ( bar_point[0][0] + bar_point[1][0] ) / 2 , bar_point[0][1] + 2])  # (+2)降低一點單槓高度 [col,row]
    # l_wrist = np.array(human_point[27:29])     # human_key_point: [col,row]
    # r_wrist = np.array(human_point[30:32])
    
    num_of_mask_point = len(human_mask)
    if num_of_mask_point == 0:
        return False
    upper_bar = 0
    
    for i in human_mask:
        if i[1] < bar_highest_point[1]:                          # 人體Mask高度與單槓比較,看大多數人體是否高於槓
            upper_bar += 1
                
    if upper_bar / num_of_mask_point > 0.9:                       # most part of body over the bar
        if get_mini_distance(bar_highest_point,human_mask) > 15:  # distance between wrist and bar is long enough so hand off the bar.(15 super-parameter)
            return True
        else:
            return False                                          # Although all body over the bar, but mabye still hand on the bar.
    else:              
        # 下法也會有under bar的情況，目前尚未考慮
        return False                                       # most human_mask must under the bar, so hand on the bar.
    
    '''
    # 以下的方法前提是pose estiamtion 100% 幾乎準確,常常會辨識上下顛倒
    if upper_bar / num_of_mask_point > 0.9:                # most part of body over the bar
        l_wrist_distance = np.sqrt(np.sum(np.square(bar_highest_point-l_wrist)))  # 歐幾里得距離(手與單槓)
        r_wrist_distance = np.sqrt(np.sum(np.square(bar_highest_point-r_wrist)))
        if l_wrist_distance > 40 or r_wrist_distance > 40: # distance between wrist and bar is long enough so hand off the bar.
            return True
        else:
            return False                                   # Although all body over the bar, but mabye still hand on the bar.
    else:                                                  
        return False                                       # most human_mask must under the bar, so hand on the bar.
    '''




# 多個 Widget 組成 Layout。  Lauout 又可以互相組合。
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./scripts/new_gui.ui", self)
        # create Upload_video button
        self.Upload_video.clicked.connect(self.open_file)

        # Main canvas -> QLabel definition
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        # self.main_canvas.setMinimumSize(100, 100)
        self.main_canvas.setMouseTracking(True) # Required for all-time tracking

        # timeline slider
        self.tl_slider.valueChanged.connect(self.tl_slide)  # 只要被拖拉，或是在城市中被set_value就會執行self.tl_slide
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        
        # some buttons
        self.playBtn.setEnabled(False)
        self.playBtn.clicked.connect(self.on_play)

        # playing flag
        self.playing_flag = False
        self.curr_frame_dirty = False

        # Create label to display current position
        self.position_label.setText('Position: 0')

        # 紀錄播放到的幀數
        self.cursur = 0
        
        # button
        self.right_one_frame.setEnabled(False)
        self.right_one_frame.clicked.connect(self.on_next_frame)
        self.left_one_frame.setEnabled(False)
        self.left_one_frame.clicked.connect(self.on_prev_frame)
        

        # motion/action data
        self.mask_data = defaultlist()    # mask for every frame(Image)
        self.human_mask_data = defaultdict(list) # {num of frame :  points of mask (coordinate)}  
        self.horizontal_bar_points = defaultlist()  # [highest point(row,col) , lowest point] of mask of bar (coordinate)
        self.pose_data = defaultlist()    # keypoint bbox for every frame
        self.twist_data = defaultlist()   # speed of loop motion of every frame
        
        # cursur timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        #plot height
        self.butt_height_data = [] # butt height of every frame
        self.height_pg.setLabel('bottom','Time','s')
        self.height_pg.showGrid(x = True, y = True, alpha = 1) 
        self.height_curve = self.height_pg.plot(self.butt_height_data) # for indicate
        self.height_pg_timer = pg.QtCore.QTimer()
        self.height_pg_timer.timeout.connect(self.update_height)
    
        # twist pg
        self.data1 = np.random.normal(size=300)
        self.twist_pg.showGrid(x = True, y = True, alpha = 1) 
        self.twist_pg.setLabel('bottom','Time','s')
        self.curve1 = self.twist_pg.plot(self.data1)
        self.twist_pg_timer = pg.QtCore.QTimer()
        self.twist_pg_timer.timeout.connect(self.update_twist)
        
        # handoff pg
        self.handoff_data = []    # true if hands on bars , false if hands not on bars for every frame.
        self.hand_off_pg.setLabel('bottom','Time','s')
        self.hand_off_pg.showGrid(x = True, y = True, alpha = 1)
        self.hand_off_curve = self.hand_off_pg.plot(self.handoff_data) # for indicate
        self.hand_off_timer = pg.QtCore.QTimer()
        self.hand_off_timer.timeout.connect(self.update_hand_off)

        # imu pg
        self.gyrox_data = imu_data_gyrox[0:300]
        self.sensors_pg.setLabel('bottom','Time','s')
        self.gyrox = self.sensors_pg.plot(self.gyrox_data)
        self.imu_pg_timer = pg.QtCore.QTimer()
        self.imu_pg_timer.timeout.connect(self.update_imu)
        
    def open_file(self):
        global imu_data_pd,imu_data_gyrox,imu_data_gyroy,imu_data_gyroz
        global imu_data_accx,imu_data_accy,imu_data_accz
        global imu_data_haccx,imu_data_haccy,imu_data_haccz, imu_data_len
        global imu_data_left,imu_data_right
        imu_data_left = 0
        imu_data_right = 1200

        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        video_name = self.filename.split('x')[-1]
        self.video_name = video_name.split('.')[0]
        self.mask_path = os.path.join('/home/m11002125/AlphaPose-1/workspace/',self.video_name, "masks/") 
        self.motion_json_path = self.filename.replace('AlphaPose_','') + '.json'
        
        # read pose data
        with open(self.motion_json_path) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
            self.pose_data = json.load(f)
        
        # load mask to self.mask_data
        self.load_mask()
    
        # imu_data_pd = pd.read_csv(self.filename+'.csv')
        # imu_data_gyrox = list(imu_data_pd['GyroX'])
        # imu_data_gyroy = list(imu_data_pd['GyroY'])
        # imu_data_gyroz = list(imu_data_pd['GyroZ'])
        # imu_data_accx  = list(imu_data_pd['AccX'])
        # imu_data_accy  = list(imu_data_pd['AccY'])
        # imu_data_accz  = list(imu_data_pd['AccZ'])
        # imu_data_haccx = list(imu_data_pd['HAccX'])
        # imu_data_haccy = list(imu_data_pd['HAccY'])
        # imu_data_haccz = list(imu_data_pd['HAccZ'])
        # imu_data_len = len(imu_data_gyrox)

        # self.json_file = self.filename+'.json'
        # self.jsonfile = self.json_file.split('/')[-1]
        # self.jsonfile = self.jsonfile.replace('AlphaPose_', '')

        # read all video and save every frame in stack
        stream = cv2.VideoCapture(self.filename)                    # 影像路徑
        self.num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))          # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps = stream.get(cv2.CAP_PROP_FPS)                     # 查看 FPS
        print('fps:',self.fps)
        print('num_of_frames:',self.num_frames)
        self.w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))          # 影片寬
        self.h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 影片長
        video_frame = []
        counter = 0
        # pdb.set_trace()
        pixel_height_ratio = 280 / (self.horizontal_bar_points[1][1] - self.horizontal_bar_points[0][1])
        
        while(stream.isOpened()):
            # print('frame:',counter)
            _, frame = stream.read()
            if frame is None:
                break
            try:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame.append(frame)
                # bbox = self.pose_data[counter]['box'] # [min_col,min_row, w , h ] 
                
                # bbox_mid = [ bbox[0] + bbox[2] / 2 , bbox[1] + bbox[3] / 2 ] # [col,row]
                # self.butt_height_data.append(self.h - (bbox[1] + bbox[3] / 2)) # height of gymnast (這裡座標要改成由左下(左下)往右上)，因為高度是由下往上
                self.butt_height_data.append((self.horizontal_bar_points[1][1] - self.pose_data[counter]['keypoints'][34]) * pixel_height_ratio)
                # pdb.set_trace()
                
                if check_handoff(self.horizontal_bar_points,self.pose_data[counter]['keypoints'],self.human_mask_data[counter],counter):                                  # check if handoff
                    self.handoff_data.append(1)  # handoff
                else: 
                    self.handoff_data.append(0)  # not handoff
    
                # if bbox_mid[1] > self.horizontal_bar_points[0][1]: # 如果 bbox中心點的row大於單槓的max_row，在圖片中看起來就是受試者在單槓下方
                #     self.handoff_data.append(0)
                # else:                                              # 如果 bbox中心點的row小於單槓的max_row，在圖片中看起來就是受試者在單槓上方
                #     self.handoff_data.append(1)
                
                counter +=1
            except:
                video_frame.append(frame)
                break
                
        self.frames = np.stack(video_frame, axis=0)               # self.frames 儲存影片的所有幀

        self.position_label.setText('Position:0/{}'.format(self.num_frames))
        # bytesPerLine = 3 * self.w

        # set timeline slider
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(self.num_frames-1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        self.tl_slider.setTickInterval(1)
        self.playBtn.setEnabled(True)

        # gui timer 再讀完影片之後才啟動
        self.height_pg_timer.start(50) # 50ms
        self.twist_pg_timer.start(50) # 50ms
        self.hand_off_timer.start(50) # 50ms
        self.imu_pg_timer.start(2) # 2ms  imu: 400Hz
        
        self.right_one_frame.setEnabled(True)
        self.left_one_frame.setEnabled(True)

    def update_interact_vis(self):
        frame = self.frames[self.cursur]
        if frame is not None:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            # self.setPixmap(QPixmap.fromImage(image))  #QLabel
            self.main_canvas.setPixmap(QPixmap(image.scaled(self.main_canvas.size(),    # canvas 的顯示
                Qt.KeepAspectRatio, Qt.FastTransformation)))
        # 播放的時候，也要更新slider cursur
        self.tl_slider.setValue(self.cursur)
    
    def tl_slide(self): # 只要slider(cursur)一改變，就要改變顯示的幀數，不管事正常播放或是拖拉
        self.cursur = self.tl_slider.value()  
        self.show_current_frame()
        self.show_keypoint_position()

    def show_current_frame(self):
        # Re-compute overlay and show the image
        # self.compose_current_im()
        self.update_interact_vis()
        self.position_label.setText('Position:{frame}/{all_frames}'.format(frame=int(self.cursur),all_frames=self.num_frames))
        self.tl_slider.setValue(self.cursur)
    
    def show_keypoint_position(self):
        
        l_eye_x,l_eye_y = self.pose_data[self.cursur]['keypoints'][3:5]
        l_eye_x,l_eye_y = int(l_eye_x),int(l_eye_y)
        str_ = f"{l_eye_x},{l_eye_y}"
        self.l_eye_acc.setText(str_)
        r_eye_x,r_eye_y = self.pose_data[self.cursur]['keypoints'][6:8]
        r_eye_x,r_eye_y = int(r_eye_x),int(r_eye_y)
        str_ = f"{r_eye_x},{r_eye_y}"
        self.r_eye_acc.setText(str_)
        
        l_ear_x,l_ear_y = self.pose_data[self.cursur]['keypoints'][9:11]
        l_ear_x,l_ear_y = int(l_ear_x),int(l_ear_y)
        str_ = f"{l_ear_x},{l_ear_y}"
        self.l_ear_acc.setText(str_)
        r_ear_x,r_ear_y = self.pose_data[self.cursur]['keypoints'][12:14]
        r_ear_x,r_ear_y = int(r_ear_x),int(r_ear_y)
        str_ = f"{r_ear_x},{r_ear_y}"
        self.r_ear_acc.setText(str_)
        
        l_shoulder_x,l_shoulder_y = self.pose_data[self.cursur]['keypoints'][15:17]
        l_shoulder_x,l_shoulder_y = int(l_shoulder_x),int(l_shoulder_y)
        str_ = f"{l_shoulder_x},{l_shoulder_y}"
        self.l_shoulder_acc.setText(str_)
        r_shoulder_x,r_shoulder_y = self.pose_data[self.cursur]['keypoints'][18:20]
        r_shoulder_x,r_shoulder_y = int(r_shoulder_x),int(r_shoulder_y)
        str_ = f"{r_shoulder_x},{r_shoulder_y}"
        self.r_shoulder_acc.setText(str_)
        
        l_elbow_x,l_elbow_y = self.pose_data[self.cursur]['keypoints'][21:23]
        l_elbow_x,l_elbow_y = int(l_elbow_x),int(l_elbow_y)
        str_ = f"{l_elbow_x},{l_elbow_y}"
        self.l_elbow_acc.setText(str_)
        r_elbow_x,r_elbow_y = self.pose_data[self.cursur]['keypoints'][24:26]
        r_elbow_x,r_elbow_y = int(r_elbow_x),int(r_elbow_y)
        str_ = f"{r_elbow_x},{r_elbow_y}"
        self.r_elbow_acc.setText(str_)
        
        l_wrist_x,l_wrist_y = self.pose_data[self.cursur]['keypoints'][27:29]
        l_wrist_x,l_wrist_y = int(l_wrist_x),int(l_wrist_y)
        str_ = f"{l_wrist_x},{l_wrist_y}"
        self.l_wrist_acc.setText(str_)
        r_wrist_x,r_wrist_y = self.pose_data[self.cursur]['keypoints'][30:32]
        r_wrist_x,r_wrist_y = int(r_wrist_x),int(r_wrist_y)
        str_ = f"{r_wrist_x},{r_wrist_y}"
        self.r_wrist_acc.setText(str_)
        
        l_hip_x,l_hip_y = self.pose_data[self.cursur]['keypoints'][33:35]
        l_hip_x,l_hip_y = int(l_hip_x),int(l_hip_y)
        str_ = f"{l_hip_x},{l_hip_y}"
        self.l_hip_acc.setText(str_)
        r_hip_x,r_hip_y = self.pose_data[self.cursur]['keypoints'][36:38]
        r_hip_x,r_hip_y = int(r_hip_x),int(r_hip_y)
        str_ = f"{r_hip_x},{r_hip_y}"
        self.r_hip_acc.setText(str_)
        
        l_knee_x,l_knee_y = self.pose_data[self.cursur]['keypoints'][39:41]
        l_knee_x,l_knee_y = int(l_knee_x),int(l_knee_y)
        str_ = f"{l_knee_x},{l_knee_y}"
        self.l_knee_acc.setText(str_)
        r_knee_x,r_knee_y = self.pose_data[self.cursur]['keypoints'][42:44]
        r_knee_x,r_knee_y = int(r_knee_x),int(r_knee_y)
        str_ = f"{r_knee_x},{r_knee_y}"
        self.r_knee_acc.setText(str_)
        
        l_ankle_x,l_ankle_y = self.pose_data[self.cursur]['keypoints'][45:47]
        l_ankle_x,l_ankle_y = int(l_ankle_x),int(l_ankle_y)
        str_ = f"{l_ankle_x},{l_ankle_y}"
        self.l_ankle_acc.setText(str_)
        r_ankle_x,r_ankle_y = self.pose_data[self.cursur]['keypoints'][48:50]
        r_ankle_x,r_ankle_y = int(r_ankle_x),int(r_ankle_y)
        str_ = f"{r_ankle_x},{r_ankle_y}"
        self.r_ankle_acc.setText(str_)
    
    def on_play(self):                 
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()   # timer 
            
    def on_time(self):                 # 更新 cursor
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.tl_slider.setValue(self.cursur)
        
    def update_height(self):
        data = self.butt_height_data[self.cursur:self.cursur+int(self.num_frames/self.fps)]
        self.height_curve.setData(data)
        self.height_curve.setPos(self.cursur, 0)

    def update_twist(self):
        # global data1, ptr1
        self.data1[:-1] = self.data1[1:]  # shift data in the array one sample left
                                # (see also: np.roll)
        self.data1[-1] = np.random.normal()
        # print(len(self.data1))
        self.curve1.setData(self.data1)
        
    def update_hand_off(self):
        data = self.handoff_data[self.cursur:self.cursur+int(self.num_frames/self.fps)]
        self.hand_off_curve.setData(data)
        self.hand_off_curve.setPos(self.cursur, 0)
        

    def update_imu(self):
        global imu_data_gyrox,imu_data_left,imu_data_right
        if self.cursur > hand_on_frame:
            imu_data_left = int(self.cursur * (imu_data_len/self.num_frames)) - hand_on_frame
            imu_data_right = int(self.cursur * (imu_data_len/self.num_frames)) - hand_on_frame+1200
            self.gyrox_data = imu_data_gyrox[imu_data_left:imu_data_right]
            self.gyrox.setData(self.gyrox_data)

    def rand(self,n):
        data = np.random.random(n)
        data[int(n*0.1):int(n*0.13)] += .5
        data[int(n*0.18)] += 2
        data[int(n*0.1):int(n*0.13)] *= 5
        data[int(n*0.18)] *= 20
        data *= 1e-12
        return data, np.arange(n, n+len(data)) / float(n)
    
    def load_mask(self):
        fnames = sorted(glob.glob(os.path.join(self.mask_path, '*.jpg')))
        if len(fnames) == 0:
            fnames = sorted(glob.glob(os.path.join(self.mask_path, '*.png')))
        frame_list = []
        # pdb.set_trace()
        t = time.time()
        for i, fname in enumerate(fnames):
            frame_list.append(np.array(Image.open(fname).convert('RGB'), dtype=np.uint8))
        print('reading mask cost', time.time() - t)
        
        self.mask_data = np.stack(frame_list, axis=0)
        
        # create human_mask 
        # mask_data有可能為圖片路徑，假如影片太長或解析度太高，我會將影片與mask都以路徑的方式讀取，而不是一次全都讀取到ram裡，
        # 除非 RAM 很大，否則讀取影片的方式就要將影片切割成圖片再一張一張讀取   
        t = time.time()
        for index,mask in enumerate(self.mask_data):
            self.human_mask_data[index].append(np.array(find_human_mask(mask)))
        print('find_human_mask cost', time.time() - t)
        
        # create horizontal mask only for 1st frame
        self.horizontal_bar_points = find_horizontal_bar_mask(self.mask_data[0])
        
        # pdb.set_trace()
        # return frames
        
    def on_prev_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur-1) # cursor 決定第幾幀
        self.tl_slider.setValue(self.cursur)

    def on_next_frame(self):  # 移至下一幀
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur+1, self.num_frames-1)
        self.tl_slider.setValue(self.cursur)
        

        
        
        

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.setWindowTitle('圖形化介面')
    Root.show()
    sys.exit(App.exec())