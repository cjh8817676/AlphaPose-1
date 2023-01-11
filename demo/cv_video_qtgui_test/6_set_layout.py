import sys
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
        uic.loadUi("my_gui.ui", self)

        #create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create Upload_video button
        self.Upload_video.clicked.connect(self.open_file)

        # slider 
        self.slider.sliderMoved.connect(self.set_position)  # 決定media播放的帧數
        self.slider.valueChanged.connect(self.onTimeSliderValueChanged)

        #create button for stop playing
        self.stopBtn.setEnabled(False)
        self.stopBtn.clicked.connect(self.stop_video)

        #create button for playing
        self.playBtn.setEnabled(False)
        self.playBtn.clicked.connect(self.play_video)

        # Create label to display current position
        self.position_label.setText('Position: 0')

        # set video Widget to mediaPlayer widget
        self.mediaPlayer.setVideoOutput(self.videowidget)
        # media player signal
        self.mediaPlayer.stateChanged.connect(self.mediastate_changed) # play and pause
        self.mediaPlayer.positionChanged.connect(self.position_changed) # 播放進度
        self.mediaPlayer.durationChanged.connect(self.duration_changed) # 換影片，所以duration也會變

        # 紀錄播放到的幀數
        self.frame = 0

        #------------plot wave----------------------- 
        self.head_height_pg.setLabel('bottom','Time','s')
        self.head_height_pg.showGrid(x = True, y = True, alpha = 1) 

        self.data1 = np.random.normal(size=300)
        self.twist_pg.showGrid(x = True, y = True, alpha = 1) 
        self.twist_pg.setLabel('bottom','Time','s')
        self.curve1 = self.twist_pg.plot(self.data1)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update1)
        self.timer.start(50) # 50ms

        self.hand_off_pg.setLabel('bottom','Time','s')
        self.hand_off_pg.showGrid(x = True, y = True, alpha = 1)

        self.gyrox_data = imu_data_gyrox[0:300]
        self.sensors_pg.setLabel('bottom','Time','s')
        self.gyrox = self.sensors_pg.plot(self.gyrox_data)
        self.timer2 = pg.QtCore.QTimer()
        self.timer2.timeout.connect(self.update2)
        self.timer2.start(2) # 2ms  imu: 400Hz

    def open_file(self):
        global imu_data_pd,imu_data_gyrox,imu_data_gyroy,imu_data_gyroz
        global imu_data_accx,imu_data_accy,imu_data_accz
        global imu_data_haccx,imu_data_haccy,imu_data_haccz, imu_data_len
        global imu_data_left,imu_data_right
        imu_data_left = 0
        imu_data_right = 1200

        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")

        imu_data_pd = pd.read_csv(filename+'.csv')
        
        imu_data_gyrox = list(imu_data_pd['GyroX'])
        imu_data_gyroy = list(imu_data_pd['GyroY'])
        imu_data_gyroz = list(imu_data_pd['GyroZ'])
        imu_data_accx  = list(imu_data_pd['AccX'])
        imu_data_accy  = list(imu_data_pd['AccY'])
        imu_data_accz  = list(imu_data_pd['AccZ'])
        imu_data_haccx = list(imu_data_pd['HAccX'])
        imu_data_haccy = list(imu_data_pd['HAccY'])
        imu_data_haccz = list(imu_data_pd['HAccZ'])
        imu_data_len = len(imu_data_gyrox)

        self.json_file = filename+'.json'
        self.jsonfile = self.json_file.split('/')[-1]
        self.jsonfile = self.jsonfile.replace('AlphaPose_', '')

        if filename != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)
        
        stream = cv2.VideoCapture(filename)                      # 影像路徑
        self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
        print('fps:',self.fps)
        self.w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))       # 影片寬
        self.h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))      # 影片長
        
        self.position_label.setText('Position:0/{}'.format(self.datalen))
    
    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.timer2.stop()
        else:
            self.mediaPlayer.play()
            self.timer2.start(2)

    def stop_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
    
    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)
            )
        else:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )

    def position_changed(self, position):
        self.slider.setValue(position)
        # calculate the current frame number:
        # frame = position / 1000.0 * self.fps
        self.frame =  position / 1000.0 * self.fps
        self.position_label.setText('Position:{frame}/{all_frames}'.format(frame=int(self.frame),all_frames=self.datalen))
        # print(f"Current frame: {frame:.0f}")
        # print('position:',position)
 
    # 每個影片都有自己的duration(影片總時常單位:ms)，所以duration_changed的意思是換影片了。
    def duration_changed(self, duration):
        print('duration:',duration)
        self.slider.setRange(0, duration)

    # 滑鼠拖拉時間線,導致position改變，又會到position_changed函式。
    def set_position(self, position): # 傳遞的可變物件 (Mutable Object) 
        print('position:', position)
        self.mediaPlayer.setPosition(position)

    def onTimeSliderValueChanged(self, value):
        # 計算出當前位置對應的帧數
        frame =  value / 1000.0 * self.fps
        # frame = self.calculateFrame(position)
        print('value',value)
        print("Frame:", frame)

    def update1(self):
        # global data1, ptr1
        self.data1[:-1] = self.data1[1:]  # shift data in the array one sample left
                                # (see also: np.roll)
        self.data1[-1] = np.random.normal()
        # print(len(self.data1))
        self.curve1.setData(self.data1)

    def update2(self):
        global imu_data_gyrox,imu_data_left,imu_data_right
        if self.frame > hand_on_frame:
            imu_data_left = int(self.frame * (imu_data_len/self.datalen)) - hand_on_frame
            imu_data_right = int(self.frame * (imu_data_len/self.datalen)) - hand_on_frame+1200
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