#! /usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import (
    Any,
    List,
    Mapping,
    Tuple,
    Union,
)
import json
from operator import (
    concat,
    getitem,
    itemgetter,
    methodcaller,
)
from functools import (
    reduce,
    partial,
)
from itertools import (
    chain,
)
import numpy as np
from darwinutils.fn import (
    f_compose_l,
    f_compose_r,
    f_group,
    f_flip_call,
)
from darwinutils.log import get_task_logger
from darwinutils.helper import (
    get_dict_filter_function_per_keys,
)


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
    from labelme.utils import img_b64_to_arr

    with open(img_path, "r") as fd:
        img_data = json.load(fd)
    keys = set(img_data.keys()) - {"imageData", "shapes"}
    return f_compose_r(
        # f_dict_filter_func_per_AOI_keys,
        f_group(
            f_compose_r(
                itemgetter("imageData"),
                img_b64_to_arr,
            ),
            itemgetter(
                "imageHeight",
                "imageWidth",
            ),
            itemgetter("shapes"),
            # template for sub-shapes by excluding "imageData" and "shapes"
            get_dict_filter_function_per_keys(keys),
        ),
        tuple,
    )(img_data)


def _warp_transform(f_wrap, f_transform, m, mat=None, points=None, *args, **kwargs):
    if mat is not None:
        if len(args) == 0:
            if "dsize" not in kwargs:
                args = [mat.shape[:2][::-1]]
            else:
                args = [kwargs.pop("dsize")]
        mat = f_wrap(mat, m, *args, **kwargs)

    if points is not None:
        transform_kwargs = {}
        if "dst" in kwargs:
            transform_kwargs["dst"] = kwargs["dst"]
        lens = list(map(len, points))
        slices = reduce(lambda y, x: y+[slice(y[-1].stop, y[-1].stop+x)], lens, [slice(0, 0)])[1:]
        points = f_compose_r(
            # expand points array
            partial(np.concatenate, axis=0),
            # reshape to satisfy f_transform
            partial(np.expand_dims, axis=1),
            # enforce type convert to satisfy f_transform
            methodcaller("astype", np.float32),
            # fire...
            partial(f_flip_call, f_transform, m, **transform_kwargs),
            # reshape to align with normal "point"
            partial(np.squeeze, axis=1),
            # reshape to align with input
            partial(partial, getitem),
            partial(f_flip_call, map, slices),
            list,
        )(points)
    return mat, points


def warp_affine_transform(*args, **kwargs):
    """
    Affine transform the image(dense) and co-related points(sparse)
    :param args: stage to cv2.warpAffine() and cv2.transform()
    :param kwargs: stage to cv2.warpAffine() and cv2.transform()
    :return:
    """
    import cv2
    return partial(_warp_transform, cv2.warpAffine, cv2.transform)(*args, **kwargs)


def warp_perspective_transform(*args, **kwargs):
    """
    Perspective transform the image(dense) and co-related points(sparse)
    :param args: stage to cv2.warpPerspective() and cv2.perspectiveTransform()
    :param kwargs: stage to cv2.warpPerspective() and cv2.perspectiveTransform()
    :return:
    """
    import cv2
    return partial(_warp_transform, cv2.warpPerspective, cv2.perspectiveTransform)(*args, **kwargs)


def convert_bbox_to_rect(bbox):
    """
    Convert bbox format to standard 4 x points rect
    :param bbox:
    :return:
    """
    ul, dr = bbox
    return ul, (ul[0], dr[1]), dr, (dr[0], ul[1])


def convert_bounding_rect_to_bbox(bounding_rect) -> Tuple[tuple, tuple]:
    """
    Convert bounding_rect output to standard ((minx, miny), (maxx, maxy)) bbox
    :param bounding_rect: cv2.boundingRect() output type
    :return:
    """
    ul = bounding_rect[:2]
    dr = (
        ul[0] + bounding_rect[2],
        ul[1] + bounding_rect[3],
    )
    return ul, dr


def _biconvert_bboxes_with_rect(f_convert, items):
    f = f_compose_r(
        partial(partial, map),
        partial(f_compose_l, list),
    )
    return f(f(f_convert))(items)


def convert_bboxes_to_rectangles(*args, **kwargs):
    """
    Convert customized shaped bboxe array to rectangle array with same shape
    :param args:
    :param kwargs:
    :return:
    """
    return partial(_biconvert_bboxes_with_rect, convert_bbox_to_rect)(*args, **kwargs)


def convert_rectangles_to_bboxes(*args, **kwargs):
    """
    Convert customized shaped rectangle array to bbox array with same shape
    :param args:
    :param kwargs:
    :return:
    """
    import cv2
    return partial(_biconvert_bboxes_with_rect, f_compose_r(
        cv2.boundingRect,
        convert_bounding_rect_to_bbox,
    ))(*args, **kwargs)


def convert_rotated_rect_to_rect(rotated_rect) -> np.array:
    """
    Convert rotated_rect output to standard 4 x points rectangle
    :param rotated_rect: cv2.minAreaRect() output type
    :return:
    """
    import cv2

    center, size, angle = rotated_rect
    # TODO: why neg angle??? it works.
    m = cv2.getRotationMatrix2D((0, 0), -angle, 1.0)
    m[:, 2] = center
    _, (pts,) = warp_affine_transform(m, points=[np.array(size)/2.0*[
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1],
    ]])
    return pts


def crop_img_by_bboxes(mat, bboxes):
    """
    Crop image by a bounding box(rectangle)
    :param img:
    :param bbox:
    :return:
    """
    return crop_img_by_polygons(mat, list(map(convert_bbox_to_rect, bboxes)))


def mask_img_by_polygons(mat, points):
    import cv2

    mask = np.zeros(mat.shape[:2], dtype="uint8")
    _ = list(map(
        f_compose_r(
            methodcaller("astype", np.int32),
            partial(cv2.fillConvexPoly, mask, color=255),
        ),
        points,
    ))
    import json
    json.dump
    return cv2.bitwise_and(mat, mat, mask=mask)


def crop_img_by_polygons(mat, points, dsize=None):
    import math
    import cv2

    x, y, w, h = cv2.boundingRect(points)

    if dsize is None:
        center, size, angle = cv2.minAreaRect(points)

        d_width = f_compose_r(
            np.square,
            np.sum,
            math.sqrt,
            math.ceil,
        )(size)
        d_height = d_width
        dsize = [d_width, d_height]

    _y, _x = np.divide(np.subtract(dsize, [h, w]), 2).astype(np.int32)

    canvas = np.zeros([*dsize[::-1], mat.shape[2]], dtype="uint8")
    canvas[_y:_y+h, _x:_x+w] = mat[y:y+h, x:x+w]

    m_translate = np.float32([[1, 0, _x-x], [0, 1, _y-y]])
    _, (points,) = warp_affine_transform(
        m_translate,
        points=[points],
    )
    return mask_img_by_polygons(canvas, [points]), points, m_translate


def dump_shape(mat, shape, transformers, orig_dict):
    from labelme.utils import img_arr_to_b64

    keys = set(orig_dict.keys()) - {"imageData", "shapes"}
    info_dict = get_dict_filter_function_per_keys(keys)(orig_dict)

    info_dict["shapes"] = [shape]
    info_dict["imageData"] = img_arr_to_b64(mat)
    info_dict["transformPipeline"] = list(map(methodcaller("tolist"), transformers))
    return info_dict


def rotate_img(angle, mat, points, center=None, *args, **kwargs):
    import cv2
    if center is None:
        center = np.floor_divide(mat.shape[:2][::-1], 2)
    m = cv2.getRotationMatrix2D(angle, center)
    return warp_affine_transform(m, mat=mat, points=points, *args, **kwargs)


def translate_img(mat, shift, points, *args, **kwargs):
    w, h = shift
    m = np.float32([[1, 0, w], [0, 1, h]])
    return warp_affine_transform(m, mat=mat, points=points, *args, **kwargs)
