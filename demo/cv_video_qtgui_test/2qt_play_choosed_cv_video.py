import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# 多個 Widget 組成 Layout。  Lauout 又可以互相組合。

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL_video = QVBoxLayout()  # 垂直排
        self.HBL = QHBoxLayout()        # 水平排列

        #  ----------------------左半邊-------------------------
        self.FeedLabel = QLabel()
        self.VBL_video.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
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

        #   ---------------------功能區-------------------------
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        #   ----------------------------------------------------
        self.setLayout(self.HBL) #　決定佈局
        # self.mw.show()

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()
        self.Worker_wave.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self): # 22 Worker1.start() 就會呼叫這裡的程式。
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):  # QThread的結束
        self.ThreadActive = False
        self.quit()

class Worker_waveform(QThread):

    def __init__(self):
        super().__init__()
        self.VBL_waveform = QVBoxLayout()
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
        t = QtCore.QTimer()
        t.timeout.connect(self.updateData)
        t.start(50)
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
    Root.resize(1600,900)
    Root.show()
    sys.exit(App.exec())