import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent 
from PyQt5.QtMultimediaWidgets import QVideoWidget

# 多個 Widget 組成 Layout。  Lauout 又可以互相組合。

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(0,0,1600,900)
        p =self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.video_path = ''
        self.VBL_video = QVBoxLayout()  # 垂直排
        self.HBL_player_button = QHBoxLayout()
        self.HBL = QHBoxLayout()        # 水平排列

        #  ---------------------left half----------------------
        #create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        #create videowidget object
        self.videowidget = QVideoWidget()
        self.VBL_video.addWidget(self.videowidget)

        #create open button
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
                
        #create open button
        self.openBtn = QPushButton('Open Video')
        self.openBtn.clicked.connect(self.open_file)
        
 
        #create button for playing
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)
        
        #create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0,0)
        self.slider.sliderMoved.connect(self.set_position)  # 決定opencv播放的帧數

        # layout
        self.HBL_player_button.addWidget(self.openBtn)
        self.HBL_player_button.addWidget(self.playBtn)
        self.HBL_player_button.addWidget(self.slider)
        self.VBL_video.addLayout(self.HBL_player_button)
        self.VBL_video.addWidget(self.CancelBTN)
        #  -----------------------------------------------------

        #  --------------function of left half------------------
        self.mediaPlayer.setVideoOutput(self.videowidget)
        # media player signal
        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        #  -----------------------------------------------------

        #  ---------------------right half----------------------
        self.Worker_wave = Worker_waveform()
        self.Worker_wave.start()
        #   ----------------------------------------------------

        #   ----------------Layout Integration-------------------
        self.HBL.addLayout(self.VBL_video)
        self.HBL.addLayout(self.Worker_wave.VBL_waveform)
        #   ----------------------------------------------------

        self.setLayout(self.HBL) #　決定佈局

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)
 
 
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
        self.slider.setValue(position)
 
    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
 
    def set_position(self, position): # 傳遞的可變物件 (Mutable Object)
        self.mediaPlayer.setPosition(position)
 
    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())
    
    def CancelFeed(self):
        pass


class Worker_waveform(QThread):

    def __init__(self):
        super().__init__()
        self.VBL_waveform = QVBoxLayout()
        print(self.VBL_waveform.totalMaximumSize())
        # plot widget (self.pw)
        self.pw = pg.PlotWidget(name='高度(Head of Bar height)')  ## giving the plots names allows us to link their axes together
        self.VBL_waveform.addWidget(self.pw)
        self.pw2 = pg.PlotWidget(name='轉速 Twist&Turn rotation speed')
        self.VBL_waveform.addWidget(self.pw2)
        self.pw3 = pg.PlotWidget(name='滯空時間')
        self.VBL_waveform.addWidget(self.pw3)

        ## Create an empty plot curve to be filled later, set its pen
        self.p1 = self.pw.plot()
        self.p1.setPen((200,200,100))
        ## Add in some extra graphics
        rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0, 1, 5e-11))
        rect.setPen(pg.mkPen(100, 200, 100))
        self.pw.addItem(rect)

        self.pw.setLabel('left', 'Value', units='V')
        self.pw.setLabel('bottom', 'Time', units='s')
        self.pw.setXRange(0, 2)
        self.pw.setYRange(0, 1e-10)

        ## Start a timer to rapidly update the plot in self.pw
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.updateData)
        self.t.start(50)
        self.updateData()

        ## Multiple parameterized plots--we can autogenerate averages for these.
        for i in range(0, 5):
            for j in range(0, 3):
                yd, xd = self.rand(10000)
                self.pw2.plot(y=yd*(j+1), x=xd, params={'iter': i, 'val': j})

        ## Test large numbers
        curve = self.pw3.plot(np.random.normal(size=100)*1e0, clickable=True)
        curve.curve.setClickable(True)
        curve.setPen('w')  ## white pen
        curve.setShadowPen(pg.mkPen((70,70,30), width=6, cosmetic=True))
        curve.sigClicked.connect(self.clicked)

        lr = pg.LinearRegionItem([1, 30], bounds=[0,100], movable=True)
        self.pw3.addItem(lr)
        line = pg.InfiniteLine(angle=90, movable=True)
        self.pw3.addItem(line)
        line.setBounds([0,200])

    def rand(self,n):
        data = np.random.random(n)
        data[int(n*0.1):int(n*0.13)] += .5
        data[int(n*0.18)] += 2
        data[int(n*0.1):int(n*0.13)] *= 5
        data[int(n*0.18)] *= 20
        data *= 1e-12
        return data, np.arange(n, n+len(data)) / float(n)
    
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

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.setWindowTitle('圖形化介面')
    Root.show()
    sys.exit(App.exec())