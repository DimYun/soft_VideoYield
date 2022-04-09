#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import time
import numpy as np
# from skimage import measure
import help_form as hp
import GUI_form

__author__ = 'Yunovidov Dmitry Dm.Yunovidov@gmail.com'


class CameraQtCapture(QtWidgets.QWidget):
    """
    Class for video window
    """
    boxes = [0, 0]
    boxes_area = [0, 0]
    img = None

    def __init__(self):
        super(QtWidgets.QWidget, self).__init__()

        # Build windows with video
        self.video_frame = QtWidgets.QLabel()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.video_frame)
        self.video_frame.setMouseTracking(True)

    def mousePressEvent(self, eventQMouseEvent):
        if eventQMouseEvent.button() == QtCore.Qt.LeftButton:
            x_rel = self.video_frame.pos().x()
            y_rel = self.video_frame.pos().y()
            x = eventQMouseEvent.pos().x() - x_rel
            y = eventQMouseEvent.pos().y() - y_rel
            print(x, y)
            self.boxes[0] = (x, y)

    def mouseReleaseEvent(self, eventQMouseEvent):
        if eventQMouseEvent.button() == QtCore.Qt.LeftButton:
            x_rel = self.video_frame.pos().x()
            y_rel = self.video_frame.pos().y()
            x = eventQMouseEvent.pos().x() - x_rel
            y = eventQMouseEvent.pos().y() - y_rel
            print(x, y)
            self.boxes[1] = (x, y)

    def photo(self):
        # For save QImage as photo from outside
        if self.img:
            file_out_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save photo',
                                                              filter='PNG files (*.png);;BMP files (*.bmp)')
            if file_out_name:
                file_name = file_out_name[0].split('.')
                if len(file_name) < 2:
                    self.img.save(file_name[0] + '.png')
                else:
                    self.img.save(file_out_name)

    def deleteLater(self):
        # For correct exit
        super(QtWidgets.QWidget, self).deleteLater()


def video_data_to_write(aver_int, data_area):
    """
    Write data from frame to file
    :param aver_int: list with float
    :param data_area: list with float
    :return: None
    """
    # calculate average for second
    # write time
    data = str(time.localtime()) + ','  # write date and time
    for i in range(len(aver_int)):
        data += str(aver_int[i]) + ', '
    for i in range(len(data_area)):
        data += str(data[i]) + ', '
    hp.write_data(data + '\n')  # write in file


class ControlWindow(QtWidgets.QWidget):
    """
    Build and rule main window
    """
    capture = None
    show_resolution = None
    camera = None
    boxes_area = [0, 0]
    record = None
    data_color = (255,255,255)
    calibr_dist = None
    cv_video_writer = None

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.timer = QtCore.QTimer()
        self.gui = GUI_form.GUI(self)
        self.capture = None
        self.setWindowTitle('Micro_Data')
        desktop = QtWidgets.QApplication.desktop()
        x = desktop.width()
        y = desktop.height()
        self.move(x / 2.0, y / 2.0)
        self.show()

        self.gui.start_button.clicked.connect(self.start_capture)
        self.gui.record_button.clicked.connect(self.record)
        self.gui.photo_button.clicked.connect(self.photo)

    def start_capture(self):
        """
        Activate camera and frame Qt capture window
        :return: None
        """
        if self.gui.start_button.text() == 'Start':
            self.gui.start_button.setText('Stop')
            self.gui.record_button.setDisabled(False)
            self.gui.photo_button.setDisabled(False)
            self.gui.set_fps_sb.setDisabled(True)
            if not self.capture:
                self.capture = CameraQtCapture()
                # self.capture.setParent(self)
                self.capture.setWindowFlags(QtCore.Qt.Tool)
            self.set_camera(is_start=True)
            self.start_video()
            self.capture.show()
        else:
            self.gui.start_button.setText('Start')
            self.gui.record_button.setDisabled(True)
            self.gui.photo_button.setDisabled(True)
            self.gui.record_button.setText('Start Record')
            self.timer.stop()
            self.set_camera(is_start=False)
            hp.write_data(None, is_close=True)
            self.capture.deleteLater()
            self.capture = None
            # self.start_video(is_start=False)
            self.gui.set_fps_sb.setDisabled(False)

    def set_camera(self, is_start):
        """
        Start or release camera instance
        :param is_start: boolean flag for set or release camera
        :return: None
        """
        if is_start:
            self.camera = cv2.VideoCapture(self.gui.cam_sb.value())
            # Set resolution
            x, y = hp.resolution_deshifr(self.gui.cam_resol_le.text())
            self.camera.set(3, x)
            self.camera.set(4, y)
        else:
            self.camera.release()
            self.camera = None

    def start_video(self):
        """
        Set timer for fps and start video
        :return: None
        """
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(1000. / self.gui.set_fps_sb.value())  # nice

    def next_frame(self):
        """
        Show and calculate new camera frame
        :return: None
        """
        # ret, frame = self.camera.read()  # take frame from camera
        frame = cv2.imread('calibration_image.png')
        if frame is not None:
            x, y = hp.resolution_deshifr(self.gui.process_resol_le.text())
            frame = hp.resize_image(frame, x, y)
            # write average RGB on frame
            aver_R = round(np.average(frame[:, :, 2]))
            aver_G = round(np.average(frame[:, :, 1]))
            aver_B = round(np.average(frame[:, :, 0]))
            aver_int = (aver_R, aver_G, aver_B)

            if self.gui.is_gray_chb.isChecked():
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                aver_int = round(np.average(frame[:, :]), 3)
                if self.gui.auto_equalize_chb.isChecked():
                    frame = cv2.equalizeHist(frame)
                    # Bilateral (Adaptive) blur, like Canny, save borders, but slow,
                    # smooth_gray_blur number should be odd and small, like 5 - 9 - area of neighbor pixel
                    # values for smooth_gray_num2: 25 - 150 - color difference between pixel, if large, then more smooth for edges
                    # frame = cv2.GaussianBlur(frame, (5, 5), 0)
                    # frame = cv2.bilateralFilter(frame, 5, 125, 75)
                    frame = cv2.medianBlur(frame, 7)

                if self.gui.auto_edge_chb.isChecked():
                    frame, lapl_frame = hp.auto_canny(frame)

                if self.gui.is_contour.isChecked():
                    pass

            # write data and time to frame
            data_text = time.asctime()
            cv2.putText(frame, data_text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, hp.data_color, 2)
            cv2.putText(frame, "Intensity: " + str(aver_int),
                        (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # write average light pixel weight for select area on frame
            data_area = 0
            if self.gui.is_area.isChecked() and self.capture.boxes[0] and self.capture.boxes[1]:
                if np.sum(np.abs(np.array(self.capture.boxes[0]) - np.array(self.capture.boxes[1]))) < 10:
                    pass
                else:
                    x1 = self.capture.boxes[0][0]
                    x2 = self.capture.boxes[1][0]
                    y1 = self.capture.boxes[0][1]
                    y2 = self.capture.boxes[1][1]
                    x1_r = min(x1, x2)
                    x2_r = max(x1, x2)
                    y1_r = min(y1, y2)
                    y2_r = max(y1, y2)

                    data_area = [0, 0, 0]
                    if self.gui.is_gray_chb.isChecked():
                        data_area = [round(np.average(frame[y1_r:y2_r, x1_r:x2_r]), 2)]
                    else:
                        area_R = round(np.average(frame[y1_r:y2_r, x1_r:x2_r, 2]), 0)
                        area_G = round(np.average(frame[y1_r:y2_r, x1_r:x2_r, 1]), 0)
                        area_B = round(np.average(frame[y1_r:y2_r, x1_r:x2_r, 0]), 0)
                        data_area = [area_R, area_G, area_B]
                    if self.gui.auto_edge_chb.isChecked():
                        #TODO: uncomment for measure
                        pass
                        # area_image = frame[y1_r:y2_r, x1_r:x2_r]
                        # contours = measure.find_contours(area_image, 0.8)
                        # data_area = [len(contours)]

                    cv2.putText(frame, 'Data in area: ' + str(data_area),
                                (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, hp.data_color, 2)

                    cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), hp.data_color,
                                  thickness=1, lineType=8, shift=0)

            if self.gui.is_calibrate.isChecked() and self.capture.boxes[0] and self.capture.boxes[1]:
                if np.sum(np.abs(np.array(self.capture.boxes[0]) -
                                 np.array(self.capture.boxes[1]))) < 10:
                    pass
                else:
                    dist = hp.calc_dist(self.capture.boxes)
                    cv2.line(frame, self.capture.boxes[0], self.capture.boxes[1],
                             self.data_color, 2)
                    self.calibr_dist = round(self.gui.calib_sb.value() / dist, 5)

                if self.capture.boxes[0] and self.capture.boxes[1] and self.calibr_dist:
                    dist = hp.calc_dist(self.capture.boxes)
                    dist = round(self.calibr_dist * dist, 5)
                    cv2.putText(frame, 'distance: ' + str(dist) + ' mm',
                                (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, self.data_color, 2)

            if self.record == 1:  # if record video
                video_data_to_write(aver_int, data_area)
                self.out.write(frame)  # write video

            format_img = QtGui.QImage.Format_RGB888
            if self.gui.is_gray_chb.isChecked():
                format_img = QtGui.QImage.Format_Indexed8
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.capture.img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], format_img)
            pix = QtGui.QPixmap.fromImage(self.capture.img)
            self.capture.video_frame.setPixmap(pix)
        else:
            print('Frame is None!')
            return

    def record(self):
        """
        Activate or deactivate record of the video
        :return: None
        """
        if not self.capture:
            QtWidgets.QMessageBox.information(None, '', 'First start the video')
        else:
            if self.gui.record_button.text() == 'Start Record':
                if not hp.cv_video_writer:
                    name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save video',
                                                                 filter='AVI files (*.avi);;')[0]
                    if name:
                        if len(name.split('.')) < 2:
                            name += '.avi'
                        self.gui.record_button.setText('Stop Record')
                        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
                        # fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
                        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                        video_resolution = self.gui.cam_resol_le.text()
                        self.cv_video_writer = cv2.VideoWriter(
                            str(name),
                            fourcc,
                            self.fps,
                            video_resolution.split('x')
                        )
                        hp.record = 1
                        # Start csv file with video data
                        hp.write_data(name.split('.')[0])
                else:
                    self.gui.record_button.setText('Stop Record')
                    hp.record = 1
            else:
                self.record_button.setText('Start Record')
                self.cv_video_writer.release()
                self.cv_video_writer = None
                hp.write_data(None, is_close=True)

    def photo(self):
        """
        Start photo capture from capture class
        :return: None
        """
        if self.capture:
            self.capture.photo()
        else:
            pass

    def closeEvent(self, event):
        """
        Close event handler
        :param event: QtEvent
        :return: event
        """
        if self.capture:
            self.capture.close()
            # self.capture.deleteLater()
        event.accept()


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())