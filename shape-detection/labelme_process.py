#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import json
from operator import (
    concat,
    itemgetter,
)
from functools import (
    reduce,
    partial,
)
from itertools import (
    chain,
)
import numpy as np
from labelme import utils
from darwinutils.fn import (
    f_compose_r,
    f_group,
)
from imutils import *
from darwinutils.log import get_task_logger
#from darwinutils.helper import (
#    get_dict_filter_function_per_keys,
#)


logger = get_task_logger(__name__)
"""
f_dict_filter_func_per_AOI_keys = get_dict_filter_function_per_keys(
    [
        "shapes",
        "imageData",
        "imageHeight",
        "imageWidth",
    ],
)
"""


def load_labeled_img(img_path: str):
    with open(img_path, "r") as fd:
        img_data = json.load(fd)
    return f_compose_r(
        # f_dict_filter_func_per_AOI_keys,
        f_group(
            f_compose_r(
                itemgetter("imageData"),
                utils.img_b64_to_arr,
            ),
            itemgetter(
                "imageHeight",
                "imageWidth",
            ),
            itemgetter("shapes"),
        ),
        tuple,
    )(img_data)


def calc_min_area_rect(points):
    """
    计算多边形(polygon)对应最小矩形框(minAreaRect)
    :param points:
    :param rotated_rect:
    :return: cv2::RotatedRect
    """
    import cv2

    return cv2.minAreaRect(points)


def _expand_bbox_to_rect(lr, rd):
    return lr, (lr[0], rd[1]), rd, (rd[0], lr[1])


def convert_rotated_rect_to_bbox(rotated_rect):
    center, (w, h), angle = rotated_rect
    angle *= math.pi/90.0
    a = -math.sin(angle)
    b = math.cos(angle)
    _w = abs(w*b + h*a)
    _h = abs(w*a + h*b)
    return center, (_w, _h), 0.0


def rotate_img(img: np.array, angle: float, scale: float = 0.0, center = None, bboxes = None, points = None):
    """
    Rotate img and its associated bboxes, points
    :param img:
    :param angle:
    :param scale:
    :param center: rotate center
    :param bboxes:
    :param points:
    :return:
    """
    import cv2
    if center is None:
        center = img.shape[:2] / 2
    if bboxes is None:
        bboxes = []
    if points is None:
        points = []

    _points = reduce(concat, chain(points, reduce(concat, bboxes)))
    if len(points) > 0:
        m = cv2.getRotationMatrix2D(center, angle, scale)
        _points = cv2.transform(
            f_compose_r(
                np.array,
                partial(np.expand_dims, axis=1),
            )(_points),
            m,
        )
        # transformed bboxes
        _bboxes = np.array(_points[len(points):]).reshape([-1, 2, 2])
        # transformed points
        points = _points[:len(points)]
        from labelme import utils
    return points


def crop_img_by_bbox(img, bbox, bboxes = None, points = None):
    """
    Crop image by a bounding box(rectangle)
    :param img:
    :param bbox:
    :param bboxes:
    :param points:
    :return:
    """
    pass


def crop_img_by_polygon(img, polygon, is_closed = False, bboxes = None, points = None):
    pass


def crop_img_by_rect(img, rect, bboxes = None, points = None):
    pass


def scale_img(img, ratio, bboxes = None, points = None):
    pass


