import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *

class main_canvas(QLabel):
    def __init__(self, parent=None):
        super(main_canvas, self).__init__(parent)
        self.setGeometry(0,0,800,640)
        self.setSizePolicy(QSizePolicy.Expanding,
            QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)

    def _update_frame(self):
        _, frame = self._capture.read()
        if frame is not None:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            # self.setPixmap(QPixmap.fromImage(image))  #QLabel
            self.setPixmap(QPixmap(image.scaled(self.size(),    # canvas 的顯示
                Qt.KeepAspectRatio, Qt.FastTransformation))
)  #QLabel

    def play(self):
        self._capture = cv2.VideoCapture('/home/m11002125/AlphaPose-1/test_video/cat_jump.mp4')
        self._timer.start(24)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    label = main_canvas()
    label.play()
    label.show()
    sys.exit(app.exec_())