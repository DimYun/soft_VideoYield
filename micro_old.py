#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PyQt4 import QtGui, QtCore
import cv2
import time
import numpy as np

__author__ = 'dmitry'

#
# class ManipulationEventFilterBuilder(Qwt5.qplt.QObject):
#     def __init__(self, parent, status_string, status_string_wait):
#         Qwt5.qplt.QObject.__init__(self, parent)
#         self.parent = parent
#         self.status_string = status_string
#         self.status_string_wait = status_string_wait
#
#     def eventFilter(self, source, event):
#         # New event filter for objects
#         if event.type() == QtCore.QEvent.MouseMove:
#             if source is self.parent.plotCanvas:
#                 pos = event.pos()
#                 x = round(float(self.parent.spec_plot.invTransform(self.parent.spec_plot.xBottom, pos.x())), 3)
#                 y = round(float(self.parent.spec_plot.invTransform(self.parent.spec_plot.yLeft, pos.y())), 1)
#                 list_elements = []
#                 for atom in ConstSM.elements_info:
#                     for line in atom['char']:
#                         try:
#                             lineEnergy = 12.398/line['l']
#                         except KeyError as var:
#                             print str(var) + '\n'
#                             print str(atom['name']) + '\n'
#                             break
#
#                         if abs(lineEnergy-x) < 0.1 and ('K' in line['type'] or 'La1' in line['type'] or
#                                                         'LB1' in line['type']):
#                             if atom['name'] not in list_elements:
#                                 list_elements.append(atom['name'])
#
#                 show_status(self.status_string.format(x, y, list_elements))
#
#             if source is self.parent.plotCanvas and self.parent.selectionStart is not None and \
#                     not self.parent.spec_plot.zoomers[0].isEnabled():
#                 pos = event.pos()
#                 xpos = pos.x()
#                 self.parent.selectedArea = (min(xpos, self.parent.selectionStart), max(xpos, self.parent.selectionStart))
#
#                 if self.parent.selectedRectangle is not None:
#                     self.parent.selectedRectangle.detach()
#
#                 self.parent.selectedRectangle = SelectionRectangle(self.parent.selectedArea)
#                 self.parent.selectedRectangle.attach(self.parent.spec_plot)
#                 self.parent.spec_plot.replot()
#
#         if (event.type() == QtCore.QEvent.MouseButtonPress and source is self.parent.plotCanvas) \
#             and not self.parent.spec_plot.zoomers[0].isEnabled():
#             self.parent.selectionStart = event.pos().x()
#
#         if (event.type() == QtCore.QEvent.MouseButtonRelease and \
#             source is self.parent.plotCanvas and self.parent.selectionStart is \
#             not None) and not self.parent.spec_plot.zoomers[0].isEnabled():
#             pos = event.pos()
#             x1 = self.parent.spec_plot.invTransform(Qwt5.QwtPlot.xBottom, min(self.parent.selectionStart, pos.x()))
#             x1 = round(float(x1), 3)
#             x2 = self.parent.spec_plot.invTransform(Qwt5.QwtPlot.xBottom, max(self.parent.selectionStart, pos.x()))
#             x2 = round(float(x2), 3)
#             if self.parent.selectedRectangle is not None:
#                 self.parent.selectedRectangle.detach()
#
#             self.parent.spec_plot.replot()
#             self.parent.selectionStart = None
#
#             str_out = str(x1) + ' ' + str(x2)  # left + right
#             self.parent.emit(QtCore.SIGNAL('BoundCalc(QString)'), str_out)
#
#             # ConstSM.peak_bounds.append([x1, x2])
#             mark_bounds(self.parent, [x1, x2])
#
#         if event.type() == QtCore.QEvent.Leave:
#             show_status(self.status_string_wait)
#
#         return QtGui.QWidget.eventFilter(self, source, event)


class QtCapture(QtGui.QWidget):
    # Class for video proceed
    record = 0
    out = None
    img = None

    def __init__(self, devise_num):
        super(QtGui.QWidget, self).__init__()
        self.fps = 24  # default fps
        self.cap = cv2.VideoCapture(devise_num)  # select web camera
        self.video_frame = QtGui.QLabel()  # label for video
        lay = QtGui.QVBoxLayout()
        lay.setMargin(0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)

    def setFPS(self, fps):
        # for set fps from outside
        self.fps = fps

    def draw_figure(self, frame, fig_type, boxes):
        # For draw figure inside PyQt
        # TODO: built it
        if fig_type:
            pass
        else:
            dx = (boxes[0][0] - boxes[1][0]) * (boxes[0][0] - boxes[1][0])
            dy = (boxes[0][1] - boxes[1][1]) * (boxes[0][1] - boxes[1][1])
            radius = int((dx + dy) ** 0.5)
            cv2.circle(frame, boxes[0], radius, (0, 255, 1), thickness=1, lineType=8, shift=0)

    def nextFrameSlot(self, fig_type=None, boxes=((),())):
        # Work with frames from webcam
        ret, frame = self.cap.read()
        if boxes[1]:  # TODO
            self.draw_figure(frame, fig_type, boxes)

        data_text = time.asctime()
        cv2.putText(frame, data_text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2)
        aver_int = str(round(np.average(frame) / 255.0, 3))
        cv2.putText(frame, 'Average intensity: ' + aver_int,
                    (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2)

        if self.record:  # if record video
            self.out.write(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # need to write conver for PyQt, case my webcam yields frames in BGR format
        self.img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(self.img)
        self.video_frame.setPixmap(pix)

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
            file_out_name = QtGui.QFileDialog.getSaveFileName(self, 'Save photo',
                                                              filter='PNG files (*.png);;BMP files (*.bmp)')
            if file_out_name:
                file_name = file_out_name.split('.')
                if len(file_name) < 2:
                    self.img.save(file_name[0] + '.png')
                else:
                    self.img.save(file_out_name)

    def deleteLater(self):
        # For correct exit
        self.cap.release()
        if self.out:
            self.out.release()
        super(QtGui.QWidget, self).deleteLater()


class ControlWindow(QtGui.QWidget):
    # Build and rule main window
    fps = None

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.capture = None

        self.set_fps_l = QtGui.QLabel('Set FPS: ')
        self.set_fps_sb = QtGui.QSpinBox()
        self.set_fps_sb.setMaximum(60)
        self.set_fps_sb.setMinimum(1)
        self.set_fps_sb.setValue(24)

        self.cam_l = QtGui.QLabel('Select camera: ')
        self.cam_sb = QtGui.QSpinBox()
        self.cam_sb.setValue(0)
        self.cam_sb.setMinimum(0)
        self.cam_sb.setMaximum(5)

        self.start_button = QtGui.QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
        self.record_button = QtGui.QPushButton('Start Record')
        self.record_button.clicked.connect(self.record)
        self.photo_button = QtGui.QPushButton('Photo')
        self.photo_button.clicked.connect(self.take_photo)
        self.end_button = QtGui.QPushButton('Change camera')
        self.end_button.setDisabled(True)
        self.end_button.clicked.connect(self.endCapture)
        # self.end_button = QtGui.QPushButton('Stop')

        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.set_fps_l)
        vbox.addWidget(self.set_fps_sb)
        vbox.addWidget(self.cam_l)
        vbox.addWidget(self.cam_sb)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.record_button)
        vbox.addWidget(self.photo_button)
        vbox.addWidget(self.end_button)
        # vbox.addWidget(self.quit_button)
        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        desktop = QtGui.QApplication.desktop()
        x = desktop.width()
        y = desktop.height()
        self.move(x / 2.0, y / 2.0)
        self.show()

    def startCapture(self):
        self.end_button.setDisabled(False)
        if self.start_button.text() == 'Start':
            self.start_button.setText('Stop')
            self.fps = self.set_fps_sb.value()
            if not self.capture:
                cam_num = self.cam_sb.value()
                self.capture = QtCapture(cam_num)
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
            self.record_button.setText('Start Record')
            self.capture.stop()
            # self.endCapture()

    def record(self):
        if not self.capture:
            QtGui.QMessageBox.information(None, '', 'First start the video')
        else:
            if self.record_button.text() == 'Start Record':
                if not self.capture.out:
                    name = QtGui.QFileDialog.getSaveFileName(self, 'Save video',
                                                             filter='AVI files (*.avi);;')
                    print type(name)
                    if name:
                        if len(name.split('.')) < 2:
                            name += '.avi'
                        self.record_button.setText('Stop Record')
                        fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
                        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                        self.capture.out = cv2.VideoWriter(str(name), fourcc, self.fps, (640, 480))
                        self.capture.record = 1
                else:
                    self.record_button.setText('Stop Record')
                    self.capture.record = 1
            else:
                self.record_button.setText('Start Record')
                self.capture.record = 0
                self.capture.out.release()

    def endCapture(self):
        self.capture.stop()
        self.capture.deleteLater()
        self.capture = None
        self.end_button.setDisabled(True)

    def take_photo(self):
        self.capture.photo()


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())
