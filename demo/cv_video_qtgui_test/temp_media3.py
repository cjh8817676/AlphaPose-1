import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class MediaPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the media player and video widget
        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        self.media_player.setVideoOutput(self.video_widget)

        # Set up the play/pause button
        self.play_button = QPushButton(self)
        self.play_button.setText('Play')
        self.play_button.clicked.connect(self.play_pause)

        # Set up the timeline slider
        self.timeline_slider = QSlider(self)
        # self.timeline_slider.setOrientation(Qt.Horizontal)
        self.timeline_slider.sliderMoved.connect(self.set_position)

        # Set up the label that displays the current frame
        self.frame_label = QLabel(self)
        self.frame_label.setText('Frame: 0')

        # Set up the layout
        self.layout = QVBoxLayout(self)
        self.control_layout = QHBoxLayout()
        self.control_layout.addWidget(self.play_button)
        self.control_layout.addWidget(self.timeline_slider)
        self.control_layout.addWidget(self.frame_label)
        self.layout.addWidget(self.video_widget)
        self.layout.addLayout(self.control_layout)
        self.setLayout(self.layout)

    def play_pause(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText('Play')
        else:
            self.media_player.play()
            self.play_button.setText('Pause')

    def set_position(self, position):
        self.media_player.setPosition(position)

    def update_frame(self):
        self.frame_label.setText('Frame: {}'.format(self.media_player.position()))

app = QApplication(sys.argv)
player = MediaPlayer()
player.show()
sys.exit(app.exec_())
