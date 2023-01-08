import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt5 Media Player')
        self.setGeometry(100, 100, 800, 600)
        
        # Create media player and video widget
        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Create play/pause button
        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.play_clicked)
        
        # Create label to display current position
        self.position_label = QLabel(self)
        self.position_label.setText('Position: 0')
        
        # Create layout and add widgets
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.video_widget)
        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.position_label)
        self.central_widget = QWidget(self)
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
    def play_clicked(self):
        # Load media file
        media = QMediaContent(QUrl.fromLocalFile('/home/m11002125/AlphaPose-1/test_video/test_video/666097171.460032.mp4'))
        self.media_player.setMedia(media)
        
        # Start playing and update position label every 100 milliseconds
        self.media_player.play()
        self.update_position_label()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_position_label)
        self.timer.start(100)
        
    def update_position_label(self):
        # Convert position (in milliseconds) to frame number
        position = self.media_player.position()
        frame = position * 30 / 1000  # Assume 30 fps
        self.position_label.setText('Position: {}'.format(int(frame)))

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())