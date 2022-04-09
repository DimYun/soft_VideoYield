#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets

__author__ = 'Yunovidov Dmitry Dm.Yunovidov@gmail.com'


class GUI(QtWidgets.QWidget):
    # Build main window

    def __init__(self, parent):
        self.main_vl = QtWidgets.QVBoxLayout(parent)
        self.set_fps_l = QtWidgets.QLabel('Set FPS: ')
        self.set_fps_sb = QtWidgets.QSpinBox()
        self.set_fps_sb.setMaximum(60)
        self.set_fps_sb.setMinimum(1)
        self.set_fps_sb.setValue(24)
        self.main_vl.addWidget(self.set_fps_l)
        self.main_vl.addWidget(self.set_fps_sb)

        self.cam_l = QtWidgets.QLabel('Select camera: ')
        self.cam_sb = QtWidgets.QSpinBox()
        self.cam_sb.setValue(0)
        self.cam_sb.setMinimum(0)
        self.cam_sb.setMaximum(5)
        self.main_vl.addWidget(self.cam_l)
        self.main_vl.addWidget(self.cam_sb)

        self.resolution_hl = QtWidgets.QHBoxLayout()
        self.cam_resol_l = QtWidgets.QLabel('Camera resolution: ')
        self.resolution_hl.addWidget(self.cam_resol_l)
        self.cam_resol_le = QtWidgets.QLineEdit('1920x1080')
        self.resolution_hl.addWidget(self.cam_resol_le)
        self.main_vl.addLayout(self.resolution_hl)

        self.resolution_hl2 = QtWidgets.QHBoxLayout()
        self.process_resol_le = QtWidgets.QLineEdit('1280x1024')
        self.process_resol_l = QtWidgets.QLabel('Resolution to show: ')
        self.resolution_hl2.addWidget(self.process_resol_l)
        self.resolution_hl2.addWidget(self.process_resol_le)
        self.main_vl.addLayout(self.resolution_hl2)

        self.is_gray_chb = QtWidgets.QCheckBox('Grayscale & Smooth')
        self.is_gray_chb.setChecked(True)
        self.main_vl.addWidget(self.is_gray_chb)

        self.auto_equalize_chb = QtWidgets.QCheckBox('Equalize')
        self.auto_equalize_chb.setChecked(True)
        self.main_vl.addWidget(self.auto_equalize_chb)

        self.auto_edge_chb = QtWidgets.QCheckBox('Canny')
        self.auto_edge_chb.setChecked(True)
        self.main_vl.addWidget(self.auto_edge_chb)

        self.is_calibrate = QtWidgets.QCheckBox('Is Calibrate')
        self.is_calibrate.setChecked(False)
        self.main_vl.addWidget(self.is_calibrate)
        self.is_area = QtWidgets.QCheckBox('Is Area')
        self.is_area.setChecked(False)
        self.main_vl.addWidget(self.is_area)
        self.is_contour = QtWidgets.QCheckBox('Is Contour')
        self.is_contour.setChecked(False)
        self.main_vl.addWidget(self.is_contour)
        self.calib_l = QtWidgets.QLabel('Calibration distance, mm: ')
        self.calib_sb = QtWidgets.QDoubleSpinBox()
        self.calib_sb.setDecimals(3)
        self.calib_sb.setSingleStep(0.001)
        self.calib_sb.setValue(0.076)
        self.main_vl.addWidget(self.calib_l)

        self.start_button = QtWidgets.QPushButton('Start')
        self.calib_button = QtWidgets.QPushButton('Calibration')
        self.calib_button.setCheckable(True)
        self.calib_button.setDisabled(True)
        self.area_int_button = QtWidgets.QPushButton('Area Intens')
        self.area_int_button.setCheckable(True)
        self.area_int_button.setChecked(False)
        self.area_int_button.setDisabled(True)
        self.record_button = QtWidgets.QPushButton('Start Record')
        self.record_button.setDisabled(True)
        self.photo_button = QtWidgets.QPushButton('Photo')
        self.photo_button.setDisabled(True)
        self.main_vl.addWidget(self.calib_sb)
        self.main_vl.addWidget(self.start_button)
        self.main_vl.addWidget(self.calib_button)
        self.main_vl.addWidget(self.area_int_button)
        self.main_vl.addWidget(self.record_button)
        self.main_vl.addWidget(self.photo_button)
