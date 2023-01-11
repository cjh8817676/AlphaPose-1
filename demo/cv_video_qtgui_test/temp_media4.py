import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QSlider, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建标签，用于显示帧
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # 创建滑块，用于控制时间轴
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1)
        self.slider.setSingleStep(0.001)
        self.slider.valueChanged.connect(self.seek)

        # 创建垂直布局
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

        # 创建中央窗口部件，并将布局添加到其中
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 打开影片文件
        self.cap = cv2.VideoCapture('/mnt/c/mydesktop/Gymnastic_Plan/workspace/AlphaPose-1/test_video/cat_jump.mp4')

        # 获取影片的总帧数和帧率
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 将时间轴的最大值设置为影片的总时长
        self.slider.setRange(0, self.total_frames / self.fps)

        # 创建定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / self.fps)

    def update_frame(self):
        # 读取帧
        ret, frame = self.cap.read()
        # 检查是否已到达影片末尾
        if not ret:
            self.timer.stop()
            return

        # 将帧转换为 QImage 对象
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)

        # 在标签中显示 QImage 对象
        self.label.setPixmap(QPixmap.fromImage(image))

    def seek(self, value):
        # 跳转到指定帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value * self.fps)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()