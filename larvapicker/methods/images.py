"""
Defines many functions that are shared across multiple components of the LarvaRetriever project
"""

import ast
import cv2
import numpy as np

from scipy.interpolate import splev, splprep
from ..config.constants import *


def img2str(img, as_bw=False):
    if as_bw and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.array(cv2.imencode('.jpg', img)[1]).tostring()


def str2img(img_str, as_bw=False):
    img = cv2.imdecode(np.frombuffer(ast.literal_eval(img_str), np.uint8), 1)
    if as_bw and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


mouseX, mouseY = 0, 0
rect = [0, 0, 0, 0]
down = False


def onMouse(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        print('MOUSE CLICK ******* x = %d, y = %d' % (x, y))
        mouseX, mouseY = x, y


def dragMouse(event, x, y, flags, param):
    global rect, down
    if event == cv2.EVENT_LBUTTONDOWN:
        print('MOUSE DOWN ******* x = %d, y = %d' % (x, y))
        rect[0], rect[1] = x, y
        down = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if down is True:
            rect[2], rect[3] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        print('MOUSE UP ******* x = %d, y = %d' % (x, y))
        rect[2], rect[3] = x, y
        down = False


def crop_img(img, x, y, size=128):
    """
    Crops the image around the coordinates (larva_x, larvaY) with safeguards as to not exceed past original image boundaries.
    Default h, w of resulting crop is 200x200.

    :param img: image to be cropped.
    :param larva_x, larva_y: coordinates that define the center around which the crop takes place.
    :return: cropped image, coordinates of where the crop took place relative to original image
    """

    (h, w) = img.shape[:2]

    if y - size < 0:
        crop = img[0:(y + size), :]
        crop = cv2.copyMakeBorder(crop, size - y, 0, 0, 0,
                                  cv2.BORDER_CONSTANT, 0)
    elif y + size > h:
        crop = img[(y - size):h, :]
        crop = cv2.copyMakeBorder(crop, 0, y + size - h, 0, 0,
                                  cv2.BORDER_CONSTANT, 0)
    else:
        crop = img[(y - size):(y + size), :]

    if x - size < 0:
        crop = crop[:, 0:(x + size)]
        crop = cv2.copyMakeBorder(crop, 0, 0, size - x, 0,
                                  cv2.BORDER_CONSTANT, 0)
    elif x + size > w:
        crop = crop[:, (x - size):w]
        crop = cv2.copyMakeBorder(crop, 0, 0, 0, x + size - w,
                                  cv2.BORDER_CONSTANT, 0)
    else:
        crop = crop[:, (x - size):(x + size)]

    return crop


def clean_img(img, mask_idx=None, threshold=True, bg=None):
    """
    This function cleans up the image with some thresholding, eroding, dilating, and background subtraction.
    This also crops refreshed larva image before writing the images to file to save time and data space.

    :param image: image to be processed
    :param dest_file: path name of file where the image will be written
    :param imgtype: defines what type of image this is -- perimeter "p", center "c", logger "l", or else
    :param cleaner: whether to apply thresholding and dilation
    :param bg: background image to subtract out
    :return: the processed image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if threshold:
        # Applying adaptive threshold around local mean
        img_inv = 255 - img
        img_at = cv2.adaptiveThreshold(
            img_inv,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11, 20
        )
        # Erode, then dilate to remove noise
        kernel = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=np.uint8
        )
        img_er = cv2.erode(img_at, kernel)
        img_clean = cv2.dilate(img_er, kernel)

    else:
        img_clean = img.copy()

    # Applying masking for the appropriate image types based on type
    if mask_idx is not None:
        img_clean[mask_idx] = 0
    if bg is not None:
        img_clean = img_clean - bg

    return img_clean


def get_contours(img, instar):
    """
    Parses the image to look for any larvae in the perimeter based on findContours by OpenCV.

    :param image: image to be processed
    :param instar: subject larva instar stage (user input)
    :return: an array of detected larva positions and the contour details for the first larva on the list
    """

    xy_list = []
    cntr_list = []

    # img is a masked image where the background is black and larvae are white
    # Now find contours. The first returned argument is an image, which is currently unused
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Compare each contour detected to the parameters defined in LPConstants.py to check if it's actually a larva
    # You can utilize the print statements to diagnose any issues
    # Otherwise it takes up too much CLI real estate, so leave it commented out
    for c in contours:

        area = cv2.contourArea(c)
        # print('Contour: ', c, 'Moments: ', mmnts)
        if area != 0:
            if (larva_area_ranges[instar - 1][0]
                    < area <
                    larva_area_ranges[instar - 1][1]):

                mmnts = cv2.moments(c)
                # Center of contour is m10/m00, m01/m00
                # print('Larva size:', mmnts['m00'], area)
                xy_list.append(
                    np.array([(mmnts['m10'] / mmnts['m00']),
                              (mmnts['m01'] / mmnts['m00'])],
                             dtype=np.int32)
                )
                c = np.array(c)
                cntr_list.append(c[:, 0])

    if len(cntr_list) > 0:
        cntr_list = np.array(cntr_list)
    else:
        cntr_list = None

    return xy_list, cntr_list


def interpolate_contour(cntr, n=32):
    cdist = np.sqrt(np.diff(cntr[:, 0]) ** 2 + np.diff(cntr[:, 1]) ** 2)
    cdist = np.append(cdist, 1)
    clist = [cntr[cdist > 0, 0].tolist(), cntr[cdist > 0, 1].tolist()]
    try:
        cf, cp = splprep(clist, s=1)
    except TypeError:
        try:
            cf, cp = splprep(clist, s=1, k=1)
        except TypeError:
            return None
    sampler_array = np.linspace(cp.min(), cp.max(), n)
    cinter = splev(sampler_array, cf)
    cinter = np.concatenate(
        (np.array(cinter[0])[:, np.newaxis],
         np.array(cinter[1])[:, np.newaxis]),
        axis=1
    )
    return cinter


def get_available_space(img):
    """
    Parses the image to look for empty spaces in the center.
    The space is divided into a grid of 50 px squares.
    If any of the pixels within each square is white, then the square is considered 'not empty'.
    This determines where larvae will be dropped off.

    :param img: image of the center of the agar to be processed
    :return: an array of positions of all available square blocks
    """

    (h, w) = img.shape

    destinations = []

    # Divides up the center here and checks for any white pixels in each square block
    for x in range(center_size // 100):
        for y in range(center_size // 100):
            if not np.any(
                    img[(h - center_size) // 2 + y * 100:(h - center_size) // 2 + (y + 1) * 100,
                        (w - center_size) // 2 + x * 100:(w - center_size) // 2 + (x + 1) * 100] > 0
            ):
                destinations.append(
                    np.array([(w - center_size) // 2 + x * 100 + 25,
                              (h - center_size) // 2 + y * 100 + 25],
                             dtype=np.int16)
                )

    return destinations
