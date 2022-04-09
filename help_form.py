#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


__author__ = 'Yunovidov Dmitry Dm.Yunovidov@gmail.com'

record = 0  # flag if record in process
cv_video_writer = None  # type cv WideoWrite, for save videoWrite object
img = None  # for save QImage object
boxes = [(), ()]  # for save user coordinate for distance
boxes_area = [(), ()]  # for save user coordinate for area
calibr_dist = None  # for save calibration distance constant
dist_to_calib = None  # flag for distance to calibrate
dist = None  # for save calculate distance data
area_sel = None  # flag for select area
data_color = (255,255,255)  # color for input data in RGB, default (200,255,155)
data_file = None  # for file-object to save data
sec = None  # for save second
intense = np.array([0.0, 0.0, 0.0])  # for calculate average intensity
area_intense = np.array([0.0, 0.0, 0.0])  # for calculate average intensity of area
count = 0  # for calculate average
cv_vers = int(cv2.__version__[0])
headers = 'Date, all R, all G, all B, area R, area G, area B, Distance\n'


def write_data(data, is_close=False):
    global data_file
    if is_close:
        if data_file:
            data_file.close()
            data_file = None
        return

    if data_file is None:
        data_file = open(data + '.csv', 'w')
        data_file.write(headers)
    else:
        data_file.write(data)


def resolution_deshifr(resolution):
    """
    Deshfrate the resolution
    :param resolution: str like 1920x1080
    :return: list of int, like [1920, 1080]
    """
    x, y = [int(j) for j in resolution.split('x')]
    return [x, y]


def resize_image(image, x, y):
    """
    Resize image for x or y scale
    :param image: cv2 image
    :param x: x size of new image resolution
    :param y: y size of new image resolution
    :return: resize image
    """
    x, y = x / image.shape[1], y / image.shape[0]
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    return cv2.resize(image, (0, 0), fx=x, fy=y)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    laplacian = cv2.Laplacian(image, cv2.CV_8U) + 255
    cv2.imwrite('calibration_image_e.png', edged)
    cv2.imwrite('calibration_image_l.png', laplacian)
    # return the edged image
    return edged, laplacian


def calc_dist(boxes):
    # Calculate distance in 'pixels'
    dx = (boxes[0][0] - boxes[1][0]) * (boxes[0][0] - boxes[1][0])
    dy = (boxes[0][1] - boxes[1][1]) * (boxes[0][1] - boxes[1][1])
    return (dx + dy) ** 0.5