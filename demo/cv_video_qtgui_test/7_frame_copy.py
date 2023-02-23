import sys,os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaMetaData
from PyQt5.QtMultimediaWidgets import QVideoWidget
import json
from PyQt5 import uic
import pandas as pd
import pdb
file_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_path,'../../XMem-1'))
from inference.interact.interactive_utils import *


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

#---------對齊--------  (之後用演算法決定、目前手動)
hand_on_frame = 253

# 多個 Widget 組成 Layout。  Lauout 又可以互相組合。
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./new_gui.ui", self)
        # create Upload_video button
        self.Upload_video.clicked.connect(self.open_file)

        # Main canvas -> QLabel definition
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)
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
        
        # cursur timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        #plot wave
        self.head_height_pg.setLabel('bottom','Time','s')
        self.head_height_pg.showGrid(x = True, y = True, alpha = 1) 

        # twist pg
        self.data1 = np.random.normal(size=300)
        self.twist_pg.showGrid(x = True, y = True, alpha = 1) 
        self.twist_pg.setLabel('bottom','Time','s')
        self.curve1 = self.twist_pg.plot(self.data1)
        self.twist_pg_timer = pg.QtCore.QTimer()
        self.twist_pg_timer.timeout.connect(self.update1)
        self.twist_pg_timer.start(50) # 50ms

        self.hand_off_pg.setLabel('bottom','Time','s')
        self.hand_off_pg.showGrid(x = True, y = True, alpha = 1)

        # imu pg
        self.gyrox_data = imu_data_gyrox[0:300]
        self.sensors_pg.setLabel('bottom','Time','s')
        self.gyrox = self.sensors_pg.plot(self.gyrox_data)
        self.imu_pg_timer = pg.QtCore.QTimer()
        self.imu_pg_timer.timeout.connect(self.update2)
        self.imu_pg_timer.start(2) # 2ms  imu: 400Hz

    
    def open_file(self):
        global imu_data_pd,imu_data_gyrox,imu_data_gyroy,imu_data_gyroz
        global imu_data_accx,imu_data_accy,imu_data_accz
        global imu_data_haccx,imu_data_haccy,imu_data_haccz, imu_data_len
        global imu_data_left,imu_data_right
        imu_data_left = 0
        imu_data_right = 1200

        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        
        # imu_data_pd = pd.read_csv(filename+'.csv')
        
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

        # self.json_file = filename+'.json'
        # self.jsonfile = self.json_file.split('/')[-1]
        # self.jsonfile = self.jsonfile.replace('AlphaPose_', '')

        # read all video and save every frame in stack
        stream = cv2.VideoCapture(filename)                      # 影像路徑
        self.num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
        print('fps:',self.fps)
        self.w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))       # 影片寬
        self.h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))      # 影片長
        video_frame = []
        while(stream.isOpened()):
            _, frame = stream.read()
            if frame is None:
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame.append(frame)
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

        # update frame
        self.video_timer.start(1000/self.fps)

    def update_interact_vis(self):
        frame = self.frames[self.cursur]
        if frame is not None:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            # self.setPixmap(QPixmap.fromImage(image))  #QLabel
            self.main_canvas.setPixmap(QPixmap(image.scaled(self.size(),    # canvas 的顯示
                Qt.KeepAspectRatio, Qt.FastTransformation)))
        # 播放的時候，也要更新slider cursur
        self.tl_slider.setValue(self.cursur)
    
    def tl_slide(self): # 只要slider(cursur)一改變，就要改變顯示的幀數，不管事正常播放或是拖拉
        self.cursur = self.tl_slider.value()  # 更新cursur
        self.show_current_frame()

    def show_current_frame(self):
        # Re-compute overlay and show the image
        # self.compose_current_im()
        self.update_interact_vis()
        self.position_label.setText('Position:{frame}/{all_frames}'.format(frame=int(self.cursur),all_frames=self.num_frames))
        self.tl_slider.setValue(self.cursur)
        
    def on_play(self):                 
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(1 / self.fps)   # timer 
            
    def on_time(self):                 # 更新 cursor
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.tl_slider.setValue(self.cursur)

    def update1(self):
        # global data1, ptr1
        self.data1[:-1] = self.data1[1:]  # shift data in the array one sample left
                                # (see also: np.roll)
        self.data1[-1] = np.random.normal()
        # print(len(self.data1))
        self.curve1.setData(self.data1)

    def update2(self):
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

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.setWindowTitle('圖形化介面')
    Root.show()
    sys.exit(App.exec())