import sys
import time
import torch
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
from ui_display import Ui_MainWindow
from Detector2 import DetectorThread, DetectorFree, PostApi
import cv2
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.thread = {}
        self.post_api_thread = {}
        
        self.start_worker_1()
        self.start_worker_2()
        self.start_worker_3()
        self.start_worker_4()
        self.W, self.H = self.uic.img1.width() * 2, self.uic.img1.height() * 2

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
        
        painter.drawLine(0, self.H // 2 + 2, self.W, self.H // 2 + 2)
        painter.drawLine(self.W // 2 + 2, 0, self.W // 2 + 2, self.H)

    def start_worker_1(self):
        self.thread[1] = DetectorFree(index=1)
        # self.post_api_thread[1] = PostApi(index=1)
        rstp1 = "rtsp://admin:atin%402022@192.168.1.232/profile1/media.smp"
        self.thread[1].setup(rstp1)
        self.thread[1].start()
        # self.post_api_thread[1].start()
        self.thread[1].signal.connect(self.my_function)

    def start_worker_2(self):
        self.thread[2] = DetectorFree(index=2)
        # self.post_api_thread[2] = PostApi(index=2)
        rstp2 = "rtsp://admin:atin%402022@192.168.1.232/profile1/media.smp"
        self.thread[2].setup(rstp2)
        self.thread[2].start()
        # self.post_api_thread[2].start()
        self.thread[2].signal.connect(self.my_function)

    def start_worker_3(self):
        self.thread[3] = DetectorFree(index=3)
        rstp3 = "rtsp://admin:atin%402022@192.168.1.232/profile1/media.smp"
        self.thread[3].setup(rstp3)
        self.thread[3].start()
        self.thread[3].signal.connect(self.my_function)

    def start_worker_4(self):
        self.thread[4] = DetectorFree(index=4)
        # self.post_api_thread[1] = PostApi(index=1)
        rstp4 = "rtsp://admin:atin%402022@192.168.1.232/profile1/media.smp"
        self.thread[4].setup(rstp4)
        self.thread[4].start()
        # self.post_api_thread[1].start()
        self.thread[4].signal.connect(self.my_function)

    # def stop_worker_1(self):
    #     self.thread[1].stop()
    #     # self.post_api_thread[1].stop()
    #     self.uic.btn_stop_1.setEnabled(False)
    #     self.uic.btn_start_1.setEnabled(True)

    # def stop_worker_2(self):
    #     self.thread[2].stop()
    #     # self.post_api_thread[2].stop()
    #     self.uic.btn_stop_2.setEnabled(False)
    #     self.uic.btn_start_2.setEnabled(True)

    # def stop_worker_3(self):
    #     self.thread[3].stop()
    #     self.uic.btn_stop_3.setEnabled(False)
    #     self.uic.btn_start_3.setEnabled(True)

    # def stop_worker_4(self):
    #     self.thread[4].stop()
    #     self.uic.btn_stop_4.setEnabled(False)
    #     self.uic.btn_start_4.setEnabled(True)

    def my_function(self, img):
        img_c = img
        rgb_img = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        qt_img = QtGui.QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], QtGui.QImage.Format_RGB888)
        thread = self.sender().index
        if thread == 1:
            self.uic.img1.setPixmap(
                QtGui.QPixmap.fromImage(qt_img).scaled(self.uic.img1.width(), self.uic.img1.height()))
        if thread == 2:
            self.uic.img2.setPixmap(
                QtGui.QPixmap.fromImage(qt_img).scaled(self.uic.img2.width(), self.uic.img2.height()))
        if thread == 3:
            self.uic.img3.setPixmap(
                QtGui.QPixmap.fromImage(qt_img).scaled(self.uic.img3.width(), self.uic.img3.height()))
        if thread == 4:
            self.uic.img4.setPixmap(
                QtGui.QPixmap.fromImage(qt_img).scaled(self.uic.img4.width(), self.uic.img4.height()))

    def add_CAM(self):
        _translate = QtCore.QCoreApplication.translate
        rstp = self.uic.text_rstp.toPlainText()
        self.uic.list_cam1.addItem(rstp)
        self.uic.list_cam2.addItem(rstp)
        self.uic.list_cam3.addItem(rstp)
        self.uic.list_cam4.addItem(rstp)
        self.uic.text_rstp.clear()

    def del_CAM(self):
        index = self.uic.text_rstp.currentIndex()
        self.uic.list_cam1.removeItem(index)
        self.uic.list_cam2.removeItem(index)
        self.uic.list_cam3.removeItem(index)
        self.uic.list_cam4.removeItem(index)
        self.uic.text_rstp.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
