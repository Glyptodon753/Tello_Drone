import numpy as np
import sys
import threading
import time
import cv2
from keras.models import load_model
from PyQt5.QtWidgets import QWidget, QGridLayout, QApplication, QLabel
from PyQt5 import QtGui
from tello import Tello
from CAMgap import cam


class Monitor(QWidget):

    def __init__(self, drone, model):
        super(Monitor, self).__init__()
        self.drone = drone
        self.model = model
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()
        self.sending_command_thread = threading.Thread(target=self.sending_command)
        self.CAM_Map = QLabel()
        self.CAM_On_Image = QLabel()
        self.Origin = QLabel()
        self.Content = QLabel()
        self.Depth_On_Image = QLabel()
        self.Depth_Map = QLabel()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.CAM_Map, 1, 0)
        grid.addWidget(self.CAM_On_Image, 0, 0)
        grid.addWidget(self.Origin, 0, 1)
        grid.addWidget(self.Content, 1, 1)
        grid.addWidget(self.Depth_Map, 1, 2)
        grid.addWidget(self.Depth_On_Image, 0, 2)
        self.move(300, 300)
        self.setWindowTitle('Tello Monitor')
        self.show()

    def update_monitor(self, image):
        predict_class, area, heatmap, heatmap_on_image = cam(self.model, image.astype('float32') / 255.0)

        qimage01 = QtGui.QImage(image, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        qimage00 = QtGui.QImage(heatmap_on_image, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        qimage10 = QtGui.QImage(heatmap, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)

        self.Origin.setPixmap(QtGui.QPixmap(qimage01))
        self.CAM_On_Image.setPixmap(QtGui.QPixmap(qimage00))
        self.CAM_Map.setPixmap(QtGui.QPixmap(qimage10))
        self.Depth_Map.setPixmap(QtGui.QPixmap(qimage01))
        self.Depth_On_Image.setPixmap(QtGui.QPixmap(qimage01))

        all_area = image.shape[0]*image.shape[1]
        self.Content.setText("Class: {0:s}\nArea: {1:d}/{2:d} pixels\nArea rate: {3:.5f} %".format(
                             predict_class, area, all_area, area/all_area))

    def video_loop(self):
        try:
            time.sleep(1)
            self.sending_command_thread.start()
            while not self.stopEvent.is_set():
                time.sleep(0.1)
                frame = self.drone.read()
                if frame is None or frame.size == 0:
                    continue
                self.update_monitor(cv2.resize(frame, dsize=(256, 192)))
        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def sending_command(self):
        while True:
            self.drone.send_command('command')
            time.sleep(5)

    def closeEvent(self, QCloseEvent):
        print("[INFO] closing...")
        self.stopEvent.set()
        del self.drone


if __name__ == '__main__':
    app = QApplication(sys.argv)
    detection = load_model('FS.h5')
    ex = Monitor(Tello('0.0.0.0', 8889), detection)
    sys.exit(app.exec())
