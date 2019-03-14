# import the necessary packages
import sys
import os
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np


def calculate_slop(start_contour, end_contour):
    x_delta = abs(end_contour[0] - start_contour[0])
    y_delta = abs(end_contour[1] - start_contour[1])
    return np.inf if y_delta == 0 else float(x_delta) / float(y_delta)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
f_img = args["image"]
image = cv2.imread(args["image"])

#
# STEP_1: 原图缩小，于性能、精度都好
#
resized = imutils.resize(image, width=640)
ratio = image.shape[0] / float(resized.shape[0])
print("ratio = {}".format(ratio))

#
# STEP_2: 转灰度
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

#
# STEP_3: 高斯降噪
#
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

canny = cv2.Canny(blurred, 10, 200)

#
# STEP_4: 二值化
#
#thresh = cv2.threshold(blurred, 135, 255, cv2.THRESH_TOZERO)[1]
thresh = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    7,
    2,
)
cv2.imwrite(f_img+"_1_thresh.jpg", thresh)
cv2.imshow("Image", thresh)
cv2.waitKey(0)



#
# STEP_5: 找轮廓
#
# find contours in the thresholded image and initialize the
# shape detector
im2, contours, hierarchy = cv2.findContours(
    thresh.copy(),
    # cv2.RETR_TREE,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
    # cv2.CHAIN_APPROX_NONE,
)
print("{} {} {}".format(np.shape(im2), np.shape(contours), np.shape(hierarchy)))
cnts = imutils.grab_contours((im2, contours, hierarchy))
#print("{} {} {}".format(np.shape(cnts[0]), np.shape(cnts[1]), np.shape(cnts[2])))
"""
for idx in range(len(contours)):
    print(np.shape(contours[idx]))
"""

_resized1 = resized.copy()
cv2.drawContours(_resized1, contours, -1, (255,255,0), 3)
cv2.imwrite(f_img+"_2_drawCoutours.jpg", _resized1)
cv2.imshow("drawContours", _resized1)
cv2.waitKey(0)


#
# STEP_6: 轮廓单独成图（轮廓图）
#
changed_img = np.zeros(np.shape(im2), dtype=np.uint8)
for idx in range(len(contours)):
    if len(contours[idx]) > 0:
        print("Contours:{}, Hierarchy: {}".format(contours[idx].shape, hierarchy[0][idx]))
        reshape_contours = np.reshape(contours[idx], (-1, 2))
        for contour in reshape_contours:
            changed_img[contour[1], contour[0]] = 255
cv2.imwrite(f_img+"_3_changed_bef.jpg", changed_img)
cv2.imshow("drawContours_bef", changed_img)
cv2.waitKey(0)

# 试试标准Canny怎么样
can_img = imutils.auto_canny(changed_img)
cv2.imwrite(f_img+"_3.1_canny.jpg", changed_img)
cv2.imshow("canny", can_img)
cv2.waitKey(0)


#
# STEP_7: 尝试轮廓封闭
#
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
changed_img = cv2.morphologyEx(changed_img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite(f_img+"_4_closed.jpg", changed_img)
cv2.imshow("closed", changed_img)
cv2.waitKey(0)


#
# STEP_5: 找轮廓again
#
# find contours in the thresholded image and initialize the
# shape detector
im2, contours, hierarchy = cv2.findContours(
    changed_img.copy(),
    # cv2.RETR_TREE,
    cv2.RETR_EXTERNAL,
    #cv2.CHAIN_APPROX_SIMPLE,
    cv2.CHAIN_APPROX_NONE,
)
print("{} {} {}".format(np.shape(im2), np.shape(contours), np.shape(hierarchy)))
cnts = imutils.grab_contours((im2, contours, hierarchy))
#print("{} {} {}".format(np.shape(cnts[0]), np.shape(cnts[1]), np.shape(cnts[2])))

_resized2 = resized.copy()
cv2.drawContours(_resized2, contours, -1, (255,255,0), 3)
cv2.imwrite(f_img+"_2_drawCoutours2.jpg", _resized2)
cv2.imshow("drawContours2", _resized2)
cv2.waitKey(0)


#
# STEP_6: 轮廓单独成图（轮廓图）
#
changed_img2 = np.zeros(np.shape(im2), dtype=np.uint8)
for idx in range(len(contours)):
    if len(contours[idx]) > 0:
        print("Contours:{}, Hierarchy: {}".format(contours[idx].shape, hierarchy[0][idx]))
        reshape_contours = np.reshape(contours[idx], (-1, 2))
        for contour in reshape_contours:
            changed_img2[contour[1], contour[0]] = 255
cv2.imwrite(f_img+"_3_changed2_bef.jpg", changed_img2)
cv2.imshow("drawContours2_bef", changed_img)
cv2.waitKey(0)

# 试试标准Canny怎么样
can_img = imutils.auto_canny(changed_img2)
cv2.imwrite(f_img+"_3.1_canny2.jpg", changed_img2)
cv2.imshow("canny2", can_img)
cv2.waitKey(0)

sys.exit(0)

lines = cv2.HoughLines(changed_img, rho=1, theta = np.pi / 180, threshold=80, min_theta=0) #, max_theta=40) np.pi / 180
#lines = cv2.HoughLines(changed_img, 1, np.pi / 180, 150, None, 0, 0)
print(len(lines))
painted_lines = []
count = 0
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        same_line = False
        """
        for existed_line in painted_lines:
            if rho - 70 <existed_line[0] < rho+70 and theta - 5 < existed_line[1]<theta +5:
                same_line = True
                break
        """
        if same_line:
            painted_lines.append([rho, theta])
            continue
        painted_lines.append([rho, theta])
        count += 1
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, pt1, pt2, (0, 0, 255), 3)

cv2.imwrite("lines.jpg", image)
cv2.imshow("lines", image)
cv2.waitKey(0)
print(count)
print(painted_lines)
'''
lines = []
start_point = reshape_contours[0]
end_point = reshape_contours[1]

slop = calculate_slop(start_point, end_point)
for contour in reshape_contours[2:]:
    new_slop = calculate_slop(start_point, contour)
    if new_slop == slop:
        # still same line, continue
    """
    print(contour)
    resized[contour[1], contour[0]] = (255, 255, 255)
    cv2.imshow("Image", resized)
    cv2.waitKey(0)
    """

'''
