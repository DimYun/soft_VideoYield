#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import time
import numpy as np
from skimage import measure

__author__ = 'Yunovidov Dmitry Dm.Yunovidov@gmail.com'


class QtVideoEdit(QtWidgets.QWidget):
    # Class for video edit
    cap = None

    def __init__(self, video_name):
        super(QtWidgets.QWidget, self).__init__()
        self.video_l = QtGui.QLabel()
        self.video_name = video_name

        self.set_fps_l = QtWidgets.QLabel('Set FPS: ')
        self.set_fps_sb = QtWidgets.QSpinBox()
        # self.set_fps_sb.setMaximum(60)
        self.set_fps_sb.setMinimum(1)
        self.set_fps_sb.setValue(24)

        # self.calib_l = QtGui.QLabel('Calibration distance, mm: ')
        # self.calib_sb = QtGui.QDoubleSpinBox()
        # self.calib_sb.setDecimals(3)
        # self.calib_sb.setSingleStep(0.001)
        # self.calib_sb.setValue(0.076)

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.startPlay)
        # self.calib_button = QtGui.QPushButton('Calibration')
        # self.calib_button.clicked.connect(self.calibration)
        # self.calib_button.setCheckable(True)
        # self.calib_button.setDisabled(True)
        # self.area_int_button = QtGui.QPushButton('Area Intens')
        # self.area_int_button.clicked.connect(self.area_int)
        # self.area_int_button.setCheckable(True)
        # self.area_int_button.setChecked(False)
        # self.area_int_button.setDisabled(True)
        # self.clear_button = QtGui.QPushButton('Clear distance')
        # self.clear_button.clicked.connect(self.clear_d)
        # self.clear_button.setDisabled(True)
        # self.record_button = QtGui.QPushButton('Start Record')
        # self.record_button.clicked.connect(self.record)
        # self.record_button.setDisabled(True)
        # self.photo_button = QtGui.QPushButton('Photo')
        # self.photo_button.clicked.connect(self.take_photo)
        # self.photo_button.setDisabled(True)
        # self.end_button = QtGui.QPushButton('Change camera')
        # self.end_button.setDisabled(True)
        # self.end_button.clicked.connect(self.endCapture)
        # # self.end_button = QtGui.QPushButton('Stop')

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.set_fps_l)
        vbox.addWidget(self.set_fps_sb)
        vbox.addWidget(self.video_l)
        # vbox.addWidget(self.cam_l)
        # vbox.addWidget(self.cam_sb)
        # vbox.addWidget(self.calib_l)
        # vbox.addWidget(self.calib_sb)
        vbox.addWidget(self.start_button)
        # vbox.addWidget(self.calib_button)
        # vbox.addWidget(self.area_int_button)
        # vbox.addWidget(self.clear_button)
        # vbox.addWidget(self.record_button)
        # vbox.addWidget(self.photo_button)
        # vbox.addWidget(self.end_button)
        # # vbox.addWidget(self.quit_button)
        self.setLayout(vbox)
        self.setWindowTitle('VideoPlayer')

    def startPlay(self):
        # Start or stop play video
        if self.start_button.text() == 'Start':
            if self.cap:
                pass
            else:
                self.cap = cv2.VideoCapture(str(self.video_name))
                self.fps = self.set_fps_sb.value()
            self.start_button.setText('Stop')
            self.start()
        else:
            self.start_button.setText('Start')
            self.stop()

    def nextFrameSlot(self, fig_type=None, boxes=((),())):
        # Work with frames from webcam
        ret, frame = self.cap.read()

        # need to write conver for PyQt, case my webcam yields frames in BGR format - no, but need convert
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # cv2.cv.CV_BGR2RGB)
        self.img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(self.img)
        self.video_l.setPixmap(pix)

    def start(self):
        # For start video from outside
        print('start video edit')
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)  # nice

    def stop(self):
        # For stop from outside
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        self.cap = None


class QtCapture(QtWidgets.QWidget):
    # Class for video proceed
    record = 0  # flag for start record
    out = None  # for save videoWrite object
    img = None  # for save QImage object
    boxes = [(), ()]  # for save user coordinate for distance
    boxes_area = [(), ()]  # for save user coordinate for area
    calibr_dist = None  # for save calibration distance constant
    dist_to_calib = None  # for distance to calibrate
    dist = None  # for save calculate distance data
    area_sel = None  # flag for select area
    data_color = (255,255,255)  # color for input data in RGB, default (200,255,155)
    data_file = None  # for file-object to save data
    sec = None  # for save second
    intense = np.array([0.0, 0.0, 0.0])  # for calculate average intensity
    area_intense = np.array([0.0, 0.0, 0.0])  # for calculate average intensity of area
    count = 0  # for calculate average
    cv_vers = int(cv2.__version__[0])

    def __init__(self, gui):
        super(QtWidgets.QWidget, self).__init__()
        self.gui = gui
        self.fps = 24  # default fps
        device_num = self.gui.cam_sb.value()
        self.cap = cv2.VideoCapture(device_num)  # select web camera
        # Set resolution
        x = self.gui.cam_resol_le.text()
        x, y = [int(j) for j in x.split('x')]
        self.cap.set(3, x)
        self.cap.set(4, y)

        # self.logo = cv2.imread('niuif_logo.png', -1)  # for transparent logo
        # self.logo = cv2.imread('niuif_1.png')
        # self.logo = cv2.resize(self.logo, (0, 0), fx=0.25, fy=0.25)
        # self.logo_y = self.logo.shape[0]
        # self.logo_x = self.logo.shape[1]
        # self.logo_a = self.logo[:, :, 3] / 255.0
        self.video_frame = QtWidgets.QLabel()  # label for video
        lay = QtWidgets.QVBoxLayout()
        lay.addWidget(self.video_frame)
        self.setLayout(lay)
        self.video_frame.setMouseTracking(True)

    def setFPS(self, fps):
        # for set fps from outside
        self.fps = fps

    def resize_image(self, image):
        """
        Resize image for x or y scale
        :param image: cv2 image
        :return: resize image
        """
        x = self.gui.process_resol_le.text()
        x, y = [float(i) for i in x.split('x')]
        x, y = x / image.shape[1], y / image.shape[0]  # auto-correct 1280X1024
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        return cv2.resize(image, (0, 0), fx=x, fy=y)

    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        print(v)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        print('\t', lower, upper)
        edged = cv2.Canny(image, lower, upper)
        laplacian = cv2.Laplacian(image, cv2.CV_8U) + 255
        # return the edged image
        return edged, laplacian

    def nextFrameSlot(self, fig_type=None, boxes=((),())):
        # Work with frames from webcam
        ret, frame = self.cap.read()
        if frame is not None:
            frame = self.resize_image(frame)
            # frame[:self.logo.shape[0], -self.logo.shape[1]:] = self.logo  # untransparent logo

            # for tranfparent logo
            # for c in range(0, 3):  # make artificial a-channel for cam video
            #     # x_o = frame.shape[1] - self.logo.shape[1]  # good
            #     x_o = 525  # fast
            #     y_o = 0
            #     frame[y_o:y_o + self.logo_y, x_o:x_o+self.logo_x, c] =\
            #         self.logo[:,:,c] * self.logo_a + frame[y_o:y_o+self.logo_y, x_o:x_o+self.logo_x, c] * (1.0 - self.logo_a)

            # write average RGB on frame
            # aver_R = round(np.average(frame[:, :, 2]))
            # aver_G = round(np.average(frame[:, :, 1]))
            # aver_B = round(np.average(frame[:, :, 0]))
            # aver_int = (aver_R, aver_G, aver_B)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.gui.auto_equalize_chb.isChecked():
                frame = cv2.equalizeHist(frame)
                # Bilateral (Adaptive) blur, like Canny, save borders, but slow,
                # smooth_gray_blur number should be odd and small, like 5 - 9 - area of neighbor pixel
                # values for smooth_gray_num2: 25 - 150 - color difference between pixel, if large, then more smooth for edges
                # frame = cv2.GaussianBlur(frame, (5, 5), 0)
                # frame = cv2.bilateralFilter(frame, 5, 125, 75)
                frame = cv2.medianBlur(frame, 5)


            if self.gui.auto_edge_chb.isChecked():
                frame, lapl_frame = self.auto_canny(frame)

            aver_int = round(np.average(frame[:, :]), 3)
            # # write average light pixel weight on frame
            # aver_int = round(np.average(frame) / 255.0, 3)

            # write data and time to frame
            data_text = time.asctime()
            cv2.putText(frame, data_text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.putText(frame, 'Int all: ' + str(aver_int),
                        (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # write average light pixel weight for select area on frame
            area_int = 0
            if self.boxes_area[0] and self.boxes_area[1]:
                if np.sum(np.abs(np.array(self.boxes_area[0]) - np.array(self.boxes_area[1]))) < 10:
                    pass
                else:
                    x1 = self.boxes_area[0][0]
                    x2 = self.boxes_area[1][0]
                    y1 = self.boxes_area[0][1]
                    y2 = self.boxes_area[1][1]
                    x1_r = min(x1, x2)
                    x2_r = max(x1, x2)
                    y1_r = min(y1, y2)
                    y2_r = max(y1, y2)

                    # need revers coordinate for cv2!
                    # area_R = round(np.average(frame[y1_r:y2_r, x1_r:x2_r, 2]), 0)
                    # area_G = round(np.average(frame[y1_r:y2_r, x1_r:x2_r, 1]), 0)
                    # area_B = round(np.average(frame[y1_r:y2_r, x1_r:x2_r, 0]), 0)
                    # area_int = (area_R, area_G, area_B)
                    # area_int = round(np.average(frame[y1_r:y2_r, x1_r:x2_r]) / 255.0, 3)
                    area_image = frame[y1_r:y2_r, x1_r:x2_r]
                    contours = measure.find_contours(area_image, 0.8)
                    num_cont = len(contours)
                    area_int = round(np.average(area_image), 3)

                    cv2.putText(frame, 'Contours in area: ' + str(num_cont),
                                (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                    cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), (255,255,255),
                                  thickness=1, lineType=8, shift=0)

        if self.boxes[0] and self.boxes[1]:
            if np.sum(np.abs(np.array(self.boxes[0]) - np.array(self.boxes[1]))) < 10:
                pass
            else:
                dist = self.calc_dist(self.boxes)
                cv2.line(frame, self.boxes[0], self.boxes[1], self.data_color, 2)
                if self.dist_to_calib:
                    if self.boxes[0] == self.boxes[1]:
                        pass
                    else:
                        self.calibr_dist = round(self.dist_to_calib / dist, 5)
                else:
                    if self.calibr_dist:
                        self.dist = round(self.calibr_dist * dist, 5)
                    # TODO: not real pxl!, solution fat, maybe
                        cv2.putText(frame, 'distance: ' + str(self.dist) + ' mm',
                                    (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, self.data_color, 2)

        if self.record:  # if record video
            aver_int = np.array(aver_int)
            area_int = np.array(area_int)
            now_time = time.localtime()
            # calculate average for second
            # print 'self.sec: ', self.sec
            if self.sec is not None:
                now_sec = now_time[5]
                if self.sec != now_sec and self.count:
                    # write time
                    data = str(data_text) + ','  # write date and time
                    # for i in now_time[:6]:
                    #     data += str(i) + ','
                    int_data = self.intense / float(self.count)
                    data += str(int_data[0]) + ', ' + str(int_data[1]) + ', ' + str(int_data[2]) + ', '
                    self.intense = np.array([0.0,0.0,0.0])
                    if self.boxes_area[0] and self.boxes_area[1]:
                        if np.sum(np.abs(np.array(self.boxes_area[0]) - np.array(self.boxes_area[1]))) > 10:
                            int_area_data = self.area_intense / float(self.count)
                            data += str(int_area_data[0]) + ', ' + str(int_area_data[1]) + ', ' + str(int_area_data[2]) + ', '
                            self.area_intense = np.array([0.0,0.0,0.0])
                        else:
                            data += 'NA, NA, NA, '
                            self.area_intense = np.array([0.0,0.0,0.0])
                    else:
                        data += 'NA, NA, NA, '
                    self.count = 1
                    if self.boxes is not None:
                        if np.sum(np.abs(np.array(self.boxes[0]) - np.array(self.boxes[1]))) > 10:
                            data += str(self.dist)
                        else:
                            data += 'NA'
                    else:
                        data += 'NA'
                    self.data_file.write(data + '\n')  # write in file
                    self.sec = now_time[5]
                else:
                    self.intense += aver_int
                    if self.boxes_area[0] and self.boxes_area[1]:
                        self.area_intense += area_int
                    self.count += 1
            else:
                self.sec = now_time[5]
                self.intense += aver_int
                if self.boxes_area[0] and self.boxes_area[1]:
                    self.area_intense += area_int
                self.count += 1

            self.out.write(frame)  # write video

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #  cv.BGR2RGB need to write conver for PyQt, case my webcam yields frames in BGR format - no, but nee convert
        self.img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_Indexed8) #Format_RGB888)
        pix = QtGui.QPixmap.fromImage(self.img)
        self.video_frame.setPixmap(pix)

    def mousePressEvent(self, eventQMouseEvent):
        if eventQMouseEvent.button() == QtCore.Qt.LeftButton:
            x = eventQMouseEvent.pos().x()
            y = eventQMouseEvent.pos().y()
            if self.area_sel:
                self.boxes_area[0] = (x, y)
            else:
                self.boxes[0] = (x, y)

    def mouseReleaseEvent(self, eventQMouseEvent):
        if eventQMouseEvent.button() == QtCore.Qt.LeftButton:
            x = eventQMouseEvent.pos().x()
            y = eventQMouseEvent.pos().y()
            if self.area_sel:
                self.boxes_area[1] = (x, y)
            else:
                self.boxes[1] = (x, y)

    def calc_dist(self, boxes):
        # Calculate distance in 'pixels'
        dx = (boxes[0][0] - boxes[1][0]) * (boxes[0][0] - boxes[1][0])
        dy = (boxes[0][1] - boxes[1][1]) * (boxes[0][1] - boxes[1][1])
        return (dx + dy) ** 0.5

    def start(self):
        # For start video from outside
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)  # nice

    def stop(self):
        # For stop from outside
        self.timer.stop()

    def photo(self):
        # For save QImage as photo from outside
        if self.img:
            file_out_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save photo',
                                                              filter='PNG files (*.png);;BMP files (*.bmp)')
            print(file_out_name)
            if file_out_name:
                file_name = file_out_name[0].split('.')
                if len(file_name) < 2:
                    self.img.save(file_name[0] + '.png')
                else:
                    self.img.save(file_out_name)

    def deleteLater(self):
        # For correct exit
        self.cap.release()
        if self.out:
            self.out.release()
        if self.data_file:
            self.data_file.close()
        super(QtWidgets.QWidget, self).deleteLater()

    def closeEvent(self, event):
        # set plot widget to normal state
        if self.cap:
            self.stop()
            self.deleteLater()
        event.accept()


class ControlWindow(QtWidgets.QWidget):
    # Build and rule main window
    fps = None

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.capture = None

        self.set_fps_l = QtWidgets.QLabel('Set FPS: ')
        self.set_fps_sb = QtWidgets.QSpinBox()
        self.set_fps_sb.setMaximum(60)
        self.set_fps_sb.setMinimum(1)
        self.set_fps_sb.setValue(24)

        self.cam_l = QtWidgets.QLabel('Select camera: ')
        self.cam_sb = QtWidgets.QSpinBox()
        self.cam_sb.setValue(0)
        self.cam_sb.setMinimum(0)
        self.cam_sb.setMaximum(5)

        self.resolution_hl = QtWidgets.QHBoxLayout()
        self.cam_resol_l = QtWidgets.QLabel('Camera resolution: ')
        self.resolution_hl.addWidget(self.cam_resol_l)
        self.cam_resol_le = QtWidgets.QLineEdit('1920x1080')
        self.resolution_hl.addWidget(self.cam_resol_le)

        self.resolution_hl2 = QtWidgets.QHBoxLayout()
        self.process_resol_le = QtWidgets.QLineEdit('1280x1024')
        self.process_resol_l = QtWidgets.QLabel('Resolution to show: ')
        self.resolution_hl2.addWidget(self.process_resol_l)
        self.resolution_hl2.addWidget(self.process_resol_le)

        self.auto_equalize_chb = QtWidgets.QCheckBox('Equalize')
        self.auto_equalize_chb.setChecked(True)

        self.auto_edge_chb = QtWidgets.QCheckBox('Canny')
        self.auto_edge_chb.setChecked(True)

        self.calib_l = QtWidgets.QLabel('Calibration distance, mm: ')
        self.calib_sb = QtWidgets.QDoubleSpinBox()
        self.calib_sb.setDecimals(3)
        self.calib_sb.setSingleStep(0.001)
        self.calib_sb.setValue(0.076)

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
        self.calib_button = QtWidgets.QPushButton('Calibration')
        self.calib_button.clicked.connect(self.calibration)
        self.calib_button.setCheckable(True)
        self.calib_button.setDisabled(True)
        self.area_int_button = QtWidgets.QPushButton('Area Intens')
        self.area_int_button.clicked.connect(self.area_int)
        self.area_int_button.setCheckable(True)
        self.area_int_button.setChecked(False)
        self.area_int_button.setDisabled(True)
        self.clear_button = QtWidgets.QPushButton('Clear distance')
        self.clear_button.clicked.connect(self.clear_d)
        self.clear_button.setDisabled(True)
        self.record_button = QtWidgets.QPushButton('Start Record')
        self.record_button.clicked.connect(self.record)
        self.record_button.setDisabled(True)
        self.photo_button = QtWidgets.QPushButton('Photo')
        self.photo_button.clicked.connect(self.take_photo)
        self.photo_button.setDisabled(True)
        self.end_button = QtWidgets.QPushButton('Change camera')
        self.end_button.setDisabled(True)
        self.end_button.clicked.connect(self.endCapture)
        self.videoEdit = QtWidgets.QPushButton('Edit video')
        self.videoEdit.clicked.connect(self.editVideo)
        # self.end_button = QtGui.QPushButton('Stop')

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.set_fps_l)
        vbox.addWidget(self.set_fps_sb)
        vbox.addWidget(self.cam_l)
        vbox.addWidget(self.cam_sb)
        vbox.addLayout(self.resolution_hl)
        vbox.addLayout(self.resolution_hl2)
        vbox.addWidget(self.auto_equalize_chb)
        vbox.addWidget(self.auto_edge_chb)
        vbox.addWidget(self.calib_l)
        vbox.addWidget(self.calib_sb)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.calib_button)
        vbox.addWidget(self.area_int_button)
        vbox.addWidget(self.clear_button)
        vbox.addWidget(self.record_button)
        vbox.addWidget(self.photo_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.videoEdit)
        # vbox.addWidget(self.quit_button)
        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        desktop = QtWidgets.QApplication.desktop()
        x = desktop.width()
        y = desktop.height()
        self.move(x / 2.0, y / 2.0)
        self.show()

    def editVideo(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open video',
                                                 filter='AVI files (*.avi);;')
        if name:
            self.capture2 = QtVideoEdit(name)
            self.capture2.show()

    def startCapture(self):
        self.end_button.setDisabled(False)
        if self.start_button.text() == 'Start':
            self.start_button.setText('Stop')
            self.calib_button.setDisabled(False)
            self.area_int_button.setDisabled(False)
            self.clear_button.setDisabled(False)
            self.record_button.setDisabled(False)
            self.photo_button.setDisabled(False)
            self.fps = self.set_fps_sb.value()
            self.set_fps_sb.setDisabled(True)
            if not self.capture:
                self.capture = QtCapture(self)
                # self.end_button.clicked.connect(self.capture.stop)
                # self.capture.setFPS(1)
                self.capture.setParent(self)
                self.capture.setFPS(self.fps)
                self.capture.setWindowFlags(QtCore.Qt.Tool)
            self.capture.setFPS(self.fps)
            self.capture.start()
            self.capture.show()
        else:
            self.start_button.setText('Start')
            self.calib_button.setDisabled(True)
            self.area_int_button.setDisabled(True)
            self.clear_button.setDisabled(True)
            self.record_button.setDisabled(True)
            self.photo_button.setDisabled(True)
            self.record_button.setText('Start Record')
            self.capture.stop()
            self.set_fps_sb.setDisabled(False)
            # self.endCapture()

    def calibration(self):
        if not self.calib_button.isChecked():
            self.capture.dist_to_calib = None
        else:
            print('start calib')
            self.capture.dist_to_calib = self.calib_sb.value()

    def area_int(self):
        # Calculate average pixel light from user selected area
        if not self.area_int_button.isChecked():
            self.capture.area_sel = None
        else:
            print('start select area')
            self.capture.area_sel = 1

    def clear_d(self):
        self.capture.boxes = [(), ()]

    def record(self):
        if not self.capture:
            QtGui.QMessageBox.information(None, '', 'First start the video')
        else:
            if self.record_button.text() == 'Start Record':
                if not self.capture.out:
                    name = QtGui.QFileDialog.getSaveFileName(self, 'Save video',
                                                             filter='AVI files (*.avi);;')
                    if name:
                        if len(name.split('.')) < 2:
                            name += '.avi'
                        self.record_button.setText('Stop Record')
                        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
                        # fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
                        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                        self.capture.out = cv2.VideoWriter(str(name), fourcc, self.fps, (640, 480))
                        self.capture.record = 1
                        name = name.split('.')[0]
                        self.capture.data_file = open(name + '.csv', 'w')
                        self.capture.data_file.write('Date, all R, all G, all B, area R, area G, area B, Distance\n')
                else:
                    self.record_button.setText('Stop Record')
                    self.capture.record = 1
            else:
                self.record_button.setText('Start Record')
                self.capture.record = 0
                self.capture.out.release()
                self.capture.data_file.close()

    def endCapture(self):
        if self.record_button.text() == 'Stop Record':
            self.record_button.setText('Start Record')
        if self.start_button.text() == 'Stop':
            self.start_button.setText('Start')
        if not self.set_fps_sb.isEnabled():
            self.set_fps_sb.setDisabled(False)
        self.capture.stop()
        self.capture.deleteLater()
        self.capture = None
        self.end_button.setDisabled(True)

    def take_photo(self):
        self.capture.photo()

    def closeEvent(self, event):
        # set plot widget to normal state
        if self.capture:
            self.capture.close()
            # self.capture.deleteLater()
        event.accept()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())