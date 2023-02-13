# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'my_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 884)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_widget = QtWidgets.QWidget(self.centralwidget)
        self.button_widget.setEnabled(True)
        self.button_widget.setGeometry(QtCore.QRect(0, 10, 931, 51))
        self.button_widget.setObjectName("button_widget")
        self.Use_camera = QtWidgets.QPushButton(self.button_widget)
        self.Use_camera.setGeometry(QtCore.QRect(160, 10, 100, 30))
        self.Use_camera.setObjectName("Use_camera")
        self.Upload_video = QtWidgets.QPushButton(self.button_widget)
        self.Upload_video.setGeometry(QtCore.QRect(10, 10, 100, 30))
        self.Upload_video.setObjectName("Upload_video")
        self.playBtn = QtWidgets.QPushButton(self.button_widget)
        self.playBtn.setGeometry(QtCore.QRect(700, 10, 100, 30))
        self.playBtn.setObjectName("playBtn")
        self.stopBtn = QtWidgets.QPushButton(self.button_widget)
        self.stopBtn.setGeometry(QtCore.QRect(820, 10, 100, 30))
        self.stopBtn.setObjectName("stopBtn")
        self.videowidget = QVideoWidget(self.centralwidget)
        self.videowidget.setGeometry(QtCore.QRect(0, 70, 931, 451))
        self.videowidget.setObjectName("videowidget")
        self.video_progress = QtWidgets.QWidget(self.centralwidget)
        self.video_progress.setGeometry(QtCore.QRect(0, 520, 931, 41))
        self.video_progress.setObjectName("video_progress")
        self.position_label = QtWidgets.QLabel(self.video_progress)
        self.position_label.setGeometry(QtCore.QRect(760, 10, 151, 16))
        self.position_label.setObjectName("position_label")
        self.slider = QtWidgets.QSlider(self.video_progress)
        self.slider.setGeometry(QtCore.QRect(10, 10, 741, 16))
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName("slider")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 560, 931, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.athelete_data = QtWidgets.QWidget(self.centralwidget)
        self.athelete_data.setGeometry(QtCore.QRect(10, 630, 401, 211))
        self.athelete_data.setObjectName("athelete_data")
        self.athelete_title = QtWidgets.QLabel(self.athelete_data)
        self.athelete_title.setGeometry(QtCore.QRect(20, 10, 300, 30))
        self.athelete_title.setMouseTracking(False)
        self.athelete_title.setObjectName("athelete_title")
        self.name = QtWidgets.QLabel(self.athelete_data)
        self.name.setGeometry(QtCore.QRect(20, 50, 101, 31))
        self.name.setObjectName("name")
        self.height = QtWidgets.QLabel(self.athelete_data)
        self.height.setGeometry(QtCore.QRect(60, 80, 51, 31))
        self.height.setObjectName("height")
        self.BMI = QtWidgets.QLabel(self.athelete_data)
        self.BMI.setGeometry(QtCore.QRect(80, 110, 31, 31))
        self.BMI.setObjectName("BMI")
        self.date_and_time = QtWidgets.QLabel(self.athelete_data)
        self.date_and_time.setGeometry(QtCore.QRect(10, 140, 101, 21))
        self.date_and_time.setObjectName("date_and_time")
        self.ip_height = QtWidgets.QLabel(self.athelete_data)
        self.ip_height.setGeometry(QtCore.QRect(120, 90, 59, 15))
        self.ip_height.setObjectName("ip_height")
        self.weifht = QtWidgets.QLabel(self.athelete_data)
        self.weifht.setGeometry(QtCore.QRect(180, 80, 51, 31))
        self.weifht.setObjectName("weifht")
        self.ip_weight = QtWidgets.QLabel(self.athelete_data)
        self.ip_weight.setGeometry(QtCore.QRect(240, 80, 59, 31))
        self.ip_weight.setObjectName("ip_weight")
        self.ip_BMI = QtWidgets.QLabel(self.athelete_data)
        self.ip_BMI.setGeometry(QtCore.QRect(120, 120, 71, 16))
        self.ip_BMI.setObjectName("ip_BMI")
        self.IP_DATE = QtWidgets.QLabel(self.athelete_data)
        self.IP_DATE.setGeometry(QtCore.QRect(120, 140, 71, 16))
        self.IP_DATE.setObjectName("IP_DATE")
        self.age = QtWidgets.QLabel(self.athelete_data)
        self.age.setGeometry(QtCore.QRect(270, 50, 51, 31))
        self.age.setObjectName("age")
        self.ip_age = QtWidgets.QLabel(self.athelete_data)
        self.ip_age.setGeometry(QtCore.QRect(330, 50, 59, 31))
        self.ip_age.setObjectName("ip_age")
        self.ip_ame = QtWidgets.QLabel(self.athelete_data)
        self.ip_ame.setGeometry(QtCore.QRect(150, 60, 59, 15))
        self.ip_ame.setObjectName("ip_ame")
        self.sensors_pg = PlotWidget(self.centralwidget)
        self.sensors_pg.setGeometry(QtCore.QRect(430, 630, 501, 221))
        self.sensors_pg.setObjectName("sensors_pg")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(410, 570, 20, 301))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.Waveform = QtWidgets.QWidget(self.centralwidget)
        self.Waveform.setGeometry(QtCore.QRect(950, 10, 641, 871))
        self.Waveform.setObjectName("Waveform")
        self.label_2 = QtWidgets.QLabel(self.Waveform)
        self.label_2.setGeometry(QtCore.QRect(290, 10, 91, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.Waveform)
        self.label_3.setGeometry(QtCore.QRect(260, 40, 151, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.Waveform)
        self.label_4.setGeometry(QtCore.QRect(140, 300, 411, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.Waveform)
        self.label_5.setGeometry(QtCore.QRect(200, 580, 251, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.head_height_pg = PlotWidget(self.Waveform)
        self.head_height_pg.setGeometry(QtCore.QRect(30, 70, 591, 221))
        self.head_height_pg.setStyleSheet("background-color: rgb(255, 255, 0);")
        self.head_height_pg.setObjectName("head_height_pg")
        self.hand_off_pg = PlotWidget(self.Waveform)
        self.hand_off_pg.setGeometry(QtCore.QRect(30, 630, 591, 221))
        self.hand_off_pg.setObjectName("hand_off_pg")
        self.twist_pg = PlotWidget(self.Waveform)
        self.twist_pg.setGeometry(QtCore.QRect(30, 340, 591, 221))
        self.twist_pg.setObjectName("twist_pg")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(930, 10, 20, 871))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(530, 590, 321, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Use_camera.setText(_translate("MainWindow", "Use Camera"))
        self.Upload_video.setText(_translate("MainWindow", "Upload Video"))
        self.playBtn.setText(_translate("MainWindow", "START"))
        self.stopBtn.setText(_translate("MainWindow", "STOP"))
        self.position_label.setText(_translate("MainWindow", "TextLabel"))
        self.athelete_title.setText(_translate("MainWindow", "                 Athllete and Video Information"))
        self.name.setText(_translate("MainWindow", "Athlete Name :"))
        self.height.setText(_translate("MainWindow", "Height :"))
        self.BMI.setText(_translate("MainWindow", "BMI:"))
        self.date_and_time.setText(_translate("MainWindow", "Date and Time:"))
        self.ip_height.setText(_translate("MainWindow", ".../cm"))
        self.weifht.setText(_translate("MainWindow", "Weight:"))
        self.ip_weight.setText(_translate("MainWindow", ".../kg"))
        self.ip_BMI.setText(_translate("MainWindow", "... kg/m^2"))
        self.IP_DATE.setText(_translate("MainWindow", "date..."))
        self.age.setText(_translate("MainWindow", "Age:"))
        self.ip_age.setText(_translate("MainWindow", ".age"))
        self.ip_ame.setText(_translate("MainWindow", "...name"))
        self.label_2.setText(_translate("MainWindow", "Parameters"))
        self.label_3.setText(_translate("MainWindow", "力道 (Strength)"))
        self.label_4.setText(_translate("MainWindow", "轉速 (TWIST & TURN ROTATION SPEED)"))
        self.label_5.setText(_translate("MainWindow", "滯空時間 (HAND OFF TIME)"))
        self.label_6.setText(_translate("MainWindow", "Vibration seneors measurements"))
from PyQt5.QtMultimediaWidgets import QVideoWidget
from pyqtgraph import PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
