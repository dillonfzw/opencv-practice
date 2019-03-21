#! /usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import (
    Any,
    Callable,
    Iterable,
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


def load_labelme_img(img_path: str) -> Mapping:
    """
    Load labelme styple image from file path
    :param img_path:
    :return:
    """
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


def dumps_labelme_shape(
        mat: np.ndarray,
        shape: Iterable[Tuple[int, int]],
        transformers: Iterable[np.ndarray],
        orig_dict: Mapping,
) -> Mapping:
    """
    Dump out the shape in an image of "labelme" type to information dictionary
    :param mat: image data to be dumped
    :param shape: image bounding polygon
    :param transformers: transform matrices which used to transform original shape to current image
    :param orig_dict: labelme typed label dict
    :return:
    """
    from labelme.utils import img_arr_to_b64

    keys = set(orig_dict.keys()) - {"imageData", "shapes"}
    info_dict = get_dict_filter_function_per_keys(keys)(orig_dict)

    # enforce converting np.array to list
    points = shape["points"]
    if isinstance(points, np.ndarray):
        points = points.tolist()
        shape["points"] = points

    info_dict["imageWidth"] = mat.shape[1]
    info_dict["imageHeight"] = mat.shape[0]

    info_dict["shapes"] = [shape]
    info_dict["imageData"] = img_arr_to_b64(mat).decode("ascii")
    # assume transformer is always np.array
    info_dict["transformPipeline"] = list(map(methodcaller("tolist"), transformers))
    return info_dict


def dump_labelme_shape(file: Union[str, Any], *args, **kwargs):
    """
    Dump out the shape in an image of "labelme" type to file
    :param file: file path or handler to be used for persistent the shape dictionary
    :param mat: image data to be dumped
    :param shape: image bounding polygon
    :param transformers: transform matrices which used to transform original shape to current image
    :param orig_dict: labelme typed label dict
    :return:
    """
    f = partial(json.dump, dumps_shape(*args, **kwargs), indent=2)

    if isinstance(file, str):
        with open(file, "w") as fd:
            f(fd)
    else:
        f(file)


def _warp_transform(
        f_wrap: Callable,
        f_transform: Callable,
        m: np.ndarray,
        mat: np.ndarray = None,
        points: Iterable[Tuple[int, int]] = None,
        *args,
        **kwargs,
) -> Tuple[
    np.ndarray,
    Iterable[Tuple[int, int]],
]:
    """
    Transform a image with specified transforming matrix
    :param f_wrap: dense/img transformer
    :param f_transform: sparse/points transformer
    :param m: transforming matrix
    :param mat: dense/image to be transformed
    :param points: sparse/points to be transformed
    :param args: other positional arguments for dense/img transformer
    :param kwargs: other named arguments for dense/img transformer
    :return: transformed image, transformed points
    """
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
    :param m: transforming matrix
    :param mat: dense/image to be transformed
    :param points: sparse/points to be transformed
    :param args: stage to cv2.warpAffine() and cv2.transform()
    :param kwargs: stage to cv2.warpAffine() and cv2.transform()
    :return:
    """
    import cv2
    return partial(_warp_transform, cv2.warpAffine, cv2.transform)(*args, **kwargs)


def warp_perspective_transform(*args, **kwargs):
    """
    Perspective transform the image(dense) and co-related points(sparse)
    :param m: transforming matrix
    :param mat: dense/image to be transformed
    :param points: sparse/points to be transformed
    :param args: stage to cv2.warpPerspective() and cv2.perspectiveTransform()
    :param kwargs: stage to cv2.warpPerspective() and cv2.perspectiveTransform()
    :return:
    """
    import cv2
    return partial(_warp_transform, cv2.warpPerspective, cv2.perspectiveTransform)(*args, **kwargs)


def convert_bbox_to_rect(bbox: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
]:
    """
    Convert bbox format to standard 4 x points rect
    :param bbox:
    :return:
    """
    ul, dr = bbox
    return ul, (ul[0], dr[1]), dr, (dr[0], ul[1])


def convert_bounding_rect_to_bbox(bounding_rect: Tuple[int, int, int, int]) -> Tuple[
    Tuple[int, int],
    Tuple[int, int],
]:
    """
    Convert bounding_rect output to standard ((minx, miny), (maxx, maxy)) bbox
    :param bounding_rect: cv2.boundingRect() output type
    :return:
    """
    pt_upleft = bounding_rect[:2]
    pt_downright = (
        pt_upleft[0] + bounding_rect[2],
        pt_upleft[1] + bounding_rect[3],
    )
    return pt_upleft, pt_downright


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


def calc_polyrect_angle(points: Iterable[Tuple[int, int]]) -> Tuple[
    float,
    Tuple[
        Tuple[int, int],
        Tuple[int, int],
        float,
    ],
    Iterable[Tuple[int, int]]
]:
    """
    Calculate the angle of a polygon described rectangle with upleft point as the first point.
    :param points:
    :return: polyrect angle, rotated rectangle struct, standard rectangle converted from rotated rectangle
    """
    import cv2
    delta_angle = [0, -90, 180, 90]

    rotated_rect = cv2.minAreaRect(points)
    center, size, angle = rotated_rect

    # 输入(外围轮廓的polygon)
    pt1 = points[0]
    # 输入(minAreaRect导出的rectangle)
    pts = convert_rotated_rect_to_rect(rotated_rect)

    distances = np.sum(np.square(pts - pt1), axis=1)
    corner_idx = np.argmin(distances)
    angle += delta_angle[corner_idx]
    return angle, rotated_rect, pts


def get_mask_from_polygons(points: Iterable[np.ndarray], dsize: Tuple[int, int]) -> np.ndarray:
    """
    Calculate a mask for a canvas with specific size from a set of polygons
    :param points:
    :param dsize: size of canvas
    :return:
    """
    import cv2

    mask = np.zeros(dsize[::-1], dtype="uint8")
    _ = list(map(
        f_compose_r(
            # convert to np.array with dtype=int32 enforcely
            # to satisfy cv2.fillConvexPoly()
            partial(np.array, dtype=np.int32),
            partial(cv2.fillConvexPoly, mask, color=255),
        ),
        points,
    ))
    return mask


def mask_img_by_polygons(mat: np.ndarray, points: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a masked image by a set of polygons
    :param mat:
    :param points:
    :return: masked image, and the mask which used to create the image for reuse
    """
    import cv2

    mask = get_mask_from_polygons(points, dsize=mat.shape[:2][::-1])
    return cv2.bitwise_and(mat, mat, mask=mask), mask


def crop_img_by_polygon(
        mat: np.ndarray,
        points: Iterable[Tuple[int, int]],
        pt_upleft: Tuple[int, int] = None,
        dsize: Tuple[int, int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop a shape from image with one polygon
    :param mat:
    :param points:
    :param pt_upleft: the up left position where the shape of target polygon will be pasted to
    :param dsize: target canvas size of cropped image
    :return: cropped image, transformed points, mask of polygons in cropped image, transform matrix
    """
    import math
    import cv2

    x, y, w, h = cv2.boundingRect(points)
    tmp_pt_upleft = (0, 0)

    # prepare canvas with a rotation-free size
    if dsize == "safe":
        _, size, angle = cv2.minAreaRect(points)

        d_width = f_compose_r(
            np.square,
            np.sum,
            math.sqrt,
            math.ceil,
        )(size)
        d_height = d_width
        dsize = (d_width, d_height)

    # minimal size to contain the shape, by default
    elif dsize == "min" or dsize is None:
        dsize = (w, h)

    # no crop, just mask with same size
    elif dsize == "same":
        dsize = mat.shape[:2][::-1]

    # paste shape to the same position by default
    if pt_upleft == "same":
        pt_upleft = (x, y)

    # the center of canvas
    elif pt_upleft == "center":
        pt_upleft = (math.ceil((dsize[0]-w)*0.5), math.ceil((dsize[1]-h)*0.5))

    # default to (0, 0)
    elif pt_upleft is None:
        pt_upleft = tmp_pt_upleft

    # prepare canvas
    canvas = np.zeros((*dsize[::-1], mat.shape[2]), dtype="uint8")

    # copy and paste shape to up-left corner of canvas
    # pad if up-left (x, y) was out size of canvas
    x_pad = max(0, -x)
    y_pad = max(0, -y)
    # pad if down-right was out size of canvas
    w_pad = min(0, dsize[0]-w)
    h_pad = min(0, dsize[1]-h)
    # TODO: not enough to cover all extream cases!!!
    canvas[y_pad:h+h_pad, x_pad:w+w_pad] = mat[y+y_pad:y+h+h_pad, x+x_pad:x+w+w_pad]
    points = np.subtract(points, [x, y])

    # shape translate to the target position in the canvas with its co-related points
    shift = np.subtract(pt_upleft, tmp_pt_upleft)
    canvas, (points,), _ = translate_img(
        shift,
        mat=canvas,
        points=[points],
    )

    # re-calculate M only for output
    shift = np.subtract(shift, [x, y])
    *_, m_translate = translate_img(shift, mat=None, points=None)
    canvas, mask = mask_img_by_polygons(canvas, [points])

    return canvas, points, mask, m_translate


def rotate_img(
        center: Tuple[int, int],
        angle: float,
        scale: float,
        mat: np.ndarray,
        points: Iterable[Tuple[int, int]],
        *args,
        **kwargs,
) -> Tuple[
    np.ndarray,
    Iterable[Tuple[int, int]],
    np.ndarray
]:
    """
    Rotate image
    :param center:
    :param angle:
    :param scale:
    :param mat:
    :param points:
    :param args:
    :param kwargs:
    :return: transformed image, transformed points, transform matrix
    """
    import cv2

    if center is None:
        center = np.floor_divide(mat.shape[:2][::-1], 2)
    if not isinstance(center, tuple):
        center = tuple(center)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    return concat(warp_affine_transform(m, mat=mat, points=points, *args, **kwargs), (m,))


def translate_img(
        shift: Tuple[int, int],
        mat: np.ndarray,
        points: Iterable[Tuple[int, int]],
        *args,
        **kwargs,
) -> Tuple[
    np.ndarray,
    Iterable[Tuple[int, int]],
    np.ndarray
]:
    """
    Translate/shift image
    :param shift:
    :param mat:
    :param points:
    :param args:
    :param kwargs:
    :return: transformed image, transformed points, transform matrix
    """
    w, h = shift
    m = np.float32([[1, 0, w], [0, 1, h]])
    return concat(warp_affine_transform(m, mat=mat, points=points, *args, **kwargs), (m,))


def paste_img(
        background: np.ndarray,
        mat: np.ndarray,
        points: Iterable[Tuple[int, int]],
        pt_upleft: Tuple[int, int] = None,
) -> Tuple[np.ndarray, Iterable[Tuple[int, int]], np.ndarray]:
    """
    Paste a shape in the image to target canvas in specified position
    :param background: background to be pasted
    :param mat: source image where the shape was located
    :param points: the polygon of target shape
    :param pt_upleft: the position where the target shape will be pasted to
    :return: manipulated image, polygons of target shape in new image, transform matrix of shape from old img to new
    """
    import cv2

    # crop mat shape to a canvas with same shape as background
    mat, points, mask, m = crop_img_by_polygon(mat, points, pt_upleft=pt_upleft, dsize=background.shape[:2][::-1])

    # clean up the target area in the background
    mask = np.bitwise_not(mask)
    background = cv2.bitwise_and(background, background, mask=mask)

    # paste the cropped mat shape to background
    canvas = cv2.bitwise_or(mat, background)
    return canvas, points, m
