import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# 多個 Widget 組成 Layout。  Lauout 又可以互相組合。

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(0,0,1600,900)

        self.video_path = ''
        self.VBL_video = QVBoxLayout()  # 垂直排
        self.HBL_player = QHBoxLayout()
        self.HBL = QHBoxLayout()        # 水平排列

        #  ----------------------左半邊-------------------------
        # indicate cv widget
        self.FeedLabel = QLabel()
        self.VBL_video.addWidget(self.FeedLabel)

        #create open button
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
                
        #create open button
        self.openBtn = QPushButton('Open Video')
        self.openBtn.clicked.connect(self.open_file)
        self.HBL_player.addWidget(self.openBtn)
 
        #create button for playing
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)
        self.HBL_player.addWidget(self.playBtn)
        #create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0,0)
        self.slider.sliderMoved.connect(self.set_position)  # 決定opencv播放的帧數
        self.HBL_player.addWidget(self.slider)

        self.VBL_video.addLayout(self.HBL_player)
        self.VBL_video.addWidget(self.CancelBTN)
        
        #  -----------------------------------------------------

        #  ----------------------右半邊-------------------------
        self.Worker_wave = Worker_waveform()
        self.Worker_wave.start()
        #   ----------------------------------------------------

        #   ---------------------畫面整合------------------------
        self.HBL.addLayout(self.VBL_video)
        self.HBL.addLayout(self.Worker_wave.VBL_waveform)
        #   ----------------------------------------------------

        self.setLayout(self.HBL) #　決定佈局
        # self.mw.show()

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def open_file(self,path):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
 
        if filename != '':
            # self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)
        self.video_path = filename
        #   ---------------------功能區-------------------------
        print(filename)
        self.Worker1 = Worker1(self.video_path)
        self.Worker1.start()    # cv play video
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            
        else:
            self.mediaPlayer.play()
        
            
    def set_position(self, position):
        self.mediaPlayer.setPosition(position)
 
        
 

    def CancelFeed(self):
        self.Worker1.stop()
        self.Worker_wave.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)  # opencv 讀取影片的每一幀，都顯示在指定的QImage上
    def __init__(self,video_path):
        super().__init__()
        self.video_path = video_path
        self.ThreadActive = True
    def run(self): # 22 Worker1.start() 就會呼叫這裡的程式。
        Capture = cv2.VideoCapture(self.video_path)  # 影像路徑
        datalen = int(Capture.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")              # Ubuntu 20.04 fourcc
        # fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        fps = Capture.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
        w = int(Capture.get(cv2.CAP_PROP_FRAME_WIDTH)) # 影片寬
        h = int(Capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 影片長
        videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': (h,w)} # 影片資訊

        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # FlippedImage = cv2.flip(Image, -1)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(800, 600, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):  # QThread的結束
        self.ThreadActive = False
        self.quit()

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