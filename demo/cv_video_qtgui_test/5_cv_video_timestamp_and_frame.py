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

# 多個 Widget 組成 Layout。  Lauout 又可以互相組合。
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(0,0,1600,900)
        # self.setStyleSheet("background-color: white;")
        # p =self.palette()
        # p.setColor(QPalette.Window, Qt.black)
        # self.setPalette(p)

        self.init_ui()

    def init_ui(self):
        self.video_path = ''
        self.VBL_video = QVBoxLayout()  # 垂直排
        self.HBL_player_button = QHBoxLayout()
        self.HBL = QHBoxLayout()        # 水平排列
        self.HBL_user_imu = QHBoxLayout()        # 水平排列
        self.HBL_NAME = QHBoxLayout()
        self.HBL_HEIGHT = QHBoxLayout()
        self.HBL_DATE = QHBoxLayout()
        self.VBL_form_and_text_box = QVBoxLayout()

        #  ---------------------left half----------------------
        #create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # self.mediaPlayer.setAspectRatioMode(Qt.KeepAspectRatio)
        
        #create videowidget object
        self.videowidget = QVideoWidget()
        self.videowidget.resize(900, 900)
        self.videowidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videowidget.setGeometry(100, 100, 1000, 1000)
        self.VBL_video.addWidget(self.videowidget)

        #create close button
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
                
        #create open button
        self.openBtn = QPushButton('Open Video')
        self.openBtn.clicked.connect(self.open_file)

        # create plotwidget for IMU
        self.pw = pg.PlotWidget(name='IMU')  ## giving the plots names allows us to link their axes together
        
        # create subject's info
        self.create_subject_info()

        #create button for playing
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)
        
        #create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        # self.slider.setMinimum(0)
        # self.slider.setMaximum(10000)
        # self.slider.setSingleStep(1)
        self.slider.sliderMoved.connect(self.set_position)  # 決定media播放的帧數
        self.slider.valueChanged.connect(self.onTimeSliderValueChanged)
        
        # Create label to display current position
        self.position_label = QLabel(self)
        self.position_label.setText('Position: 0')

        # layout
        self.HBL_player_button.addWidget(self.openBtn)
        self.HBL_player_button.addWidget(self.playBtn)
        self.HBL_player_button.addWidget(self.slider)
        self.HBL_player_button.addWidget(self.position_label)
        self.VBL_video.addLayout(self.HBL_player_button)
        self.VBL_video.addWidget(self.CancelBTN)

        self.HBL_user_imu.addLayout(self.VBL_form_and_text_box)
        self.HBL_user_imu.addWidget(self.pw)
        
        self.VBL_video.addLayout(self.HBL_user_imu)
        #  -----------------------------------------------------

        #  --------------function of left half------------------
        # set video Widget to mediaPlayer widget
        self.mediaPlayer.setVideoOutput(self.videowidget)
        # media player signal

        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        #  -----------------------------------------------------

        #  ---------------------right half----------------------
        self.Worker_wave = Worker_waveform() # just for layout here
        #   ----------------------------------------------------

        #   ----------------Layout Integration-------------------
        self.HBL.addLayout(self.VBL_video)
        self.HBL.addLayout(self.Worker_wave.VBL_waveform)
        #   ----------------------------------------------------

        self.setLayout(self.HBL) #　決定佈局
    
    def create_subject_info(self):
        # creat textbox for indicate subjects's info
        self.text_title = QLabel('Athelete and video information')
        self.text_name = QLabel('體操選手的名子')
        self.textbox_name = QLineEdit()
        self.HBL_NAME.addWidget(self.text_name)
        self.HBL_NAME.addWidget(self.textbox_name)

        self.text_height = QLabel('體操選手的身高')
        self.textbox_height = QLineEdit()
        self.HBL_HEIGHT.addWidget(self.text_height)
        self.HBL_HEIGHT.addWidget(self.textbox_height)

        self.text_date = QLabel('日期')
        self.textbox_date = QLineEdit()
        self.HBL_DATE.addWidget(self.text_date)
        self.HBL_DATE.addWidget(self.textbox_date)

        # self.VBL_form_and_text_box.addWidget(self.text_title)
        self.VBL_form_and_text_box.addLayout(self.HBL_NAME)
        self.VBL_form_and_text_box.addLayout(self.HBL_HEIGHT)
        self.VBL_form_and_text_box.addLayout(self.HBL_DATE)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        self.json_file = filename+'.json'
        self.jsonfile = self.json_file.split('/')[-1]
        self.jsonfile = self.jsonfile.replace('AlphaPose_', '')

        if filename != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)
        
        stream = cv2.VideoCapture(filename)  # 影像路徑
        self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
        self.w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) # 影片寬
        self.h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 影片長
        
        self.position_label.setText('Position:0/{}'.format(self.datalen))
 
 
    def play_video(self):
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
        # print('position cvhange:', position)
        self.slider.setValue(position)
        
        # calculate the current frame number:
        # frame = position / 1000.0 * self.fps
        frame =  position / 1000.0 * self.fps
        self.position_label.setText('Position:{frame}/{all_frames}'.format(frame=int(frame),all_frames=self.datalen))
        print(f"Current frame: {frame:.0f}")
        print('position:',position)
 
    # 每個影片都有自己的duration(影片總時常單位;/ms)，所以duration_changed的意思是換影片了。
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
 
    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())
    
    def CancelFeed(self):
        pass


class Worker_waveform(QThread):
    def __init__(self):
        super().__init__()
        
        self.VBL_waveform = QVBoxLayout()
        self.pw = pg.PlotWidget(name='高度(Head of Bar height)')  ## giving the plots names allows us to link their axes together
        self.VBL_waveform.addWidget(self.pw)
        self.pw2 = pg.PlotWidget(name='轉速 Twist&Turn rotation speed')
        self.VBL_waveform.addWidget(self.pw2)
        self.pw3 = pg.PlotWidget(name='滯空時間')
        self.VBL_waveform.addWidget(self.pw3)
        self.data1 = np.random.normal(size=300)

        self.curve1 = self.pw.plot(self.data1)

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def update(self):
        self.update1()
    
    def update1(self):
        # global data1, ptr1
        self.data1[:-1] = self.data1[1:]  # shift data in the array one sample left
                                # (see also: np.roll)
        self.data1[-1] = np.random.normal()
        # print(len(self.data1))
        self.curve1.setData(self.data1)
        
        
        # self.ptr1 += 1
        # self.pw2.setData(self.data1)
        # self.pw2.setPos(self.ptr1, 0)

    def rand(self,n):
        data = np.random.random(n)
        data[int(n*0.1):int(n*0.13)] += .5
        data[int(n*0.18)] += 2
        data[int(n*0.1):int(n*0.13)] *= 5
        data[int(n*0.18)] *= 20
        data *= 1e-12
        return data, np.arange(n, n+len(data)) / float(n)
    def high_wave(self,):
        with open(self.json_path) as f:
            content2 = json.load(f)
    
    def updateData(self):
        yd, xd = self.rand(10000)
        self.p1.setData(y=yd, x=xd)
    def run(self): 
        pg.exec()
    def clicked(self):
        print("curve clicked")
    def stop(self):   # QThread的結束
        self.ThreadActive = False
        self.quit()
    def __delete__(self):
        print('delete object')

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.setWindowTitle('圖形化介面')
    Root.show()
    sys.exit(App.exec())