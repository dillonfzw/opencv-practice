#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import nose
import math
import numpy as np
from darwinutils.fn import (
    f_compose_r,
)
from darwinutils.log import get_task_logger
from labelme_process import (
    load_labeled_img,
    calc_min_area_rect,
    convert_rotated_rect_to_bbox,
)


logger = get_task_logger(__name__)


class Test_load_labeled_img:
    def __init__(self):
        self._img_dir = f_compose_r(
            os.path.expanduser,
            os.path.expandvars,
        )("~/Pictures/data/origin")

    @property
    def img_dir(self):
        return self._img_dir

    def test_1(self):
        img_data, img_size, shapes = load_labeled_img(
            os.path.join(
                self.img_dir,
                "done",
                "invoice_ - 11.json",
            ))
        nose.tools.assert_true(
            np.all(np.equal(img_data.shape[:2], np.array(img_size))),
            "{} vs. {}".format(
                img_data.shape,
                img_size,
            ))
        nose.tools.assert_less(0, len(shapes))


class Test_calc_min_area_rect:
    def __init__(self):
        self._img_dir = f_compose_r(
            os.path.expanduser,
            os.path.expandvars,
        )("~/Pictures/data/origin")

    @property
    def img_dir(self):
        return self._img_dir

    def test_1(self):
        center, size, angle = calc_min_area_rect(np.array([
            [0, 0],
            [0, 10],
            [10, 10],
            [10, 0],
        ]))
        nose.tools.assert_equal((5, 5), center)
        nose.tools.assert_equal((10, 10), size)
        nose.tools.assert_equal(angle, -90.0)


class Test_convert_rotated_rect_to_bbox:
    def test_square_degree_45(self):
        pass
        """
        45度偏转的正方形 换算 到 旋正的 正方形
        """
        import cv2
        rotated_rect = cv2.minAreaRect(np.array([
            [2, 1],
            [3, 2],
            [2, 3],
            [1, 2],
        ]))
        center, size, angle = rotated_rect
        nose.tools.assert_equal((2.0, 2.0), center)
        nose.tools.assert_true(np.all(np.isclose(size, [math.sqrt(2), math.sqrt(2)], atol=1e-6)))
        nose.tools.assert_true(math.isclose(angle, -45.0, abs_tol=1e-6))

        nose.tools.assert_equal(
            (center, size, 0.0),
            convert_rotated_rect_to_bbox(rotated_rect),
        )

    def test_1(self):
        pass
        """
        45度偏转的长方形 换算 到 旋正的 长方形
        """
        rotated_rect = (10, 5), (5.0, 5.0/3.0*4.0), -math.asin(3.0/5.0)/math.pi*180.0
        center, size, angle = rotated_rect

        nose.tools.assert_equal(
            (center, (8.0, 5/3.0*5.0), 0.0),
            convert_rotated_rect_to_bbox(rotated_rect),
        )
