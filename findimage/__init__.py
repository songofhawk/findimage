"""
To find a template image(smaller) in a source image(bigger)

This project is inspired from https://github.com/NetEaseGame/aircv.git, which is not maintained for a long time.

There are several improvements and changes in this projects:
* support finding grayscale image, either source or template
* support finding image with transparent channel
* optimized the performance of find_all, use numpy slicing set data instead of floodFill
* removed methods that are not related to finding images
"""
__version__ = '0.1.1'

import time

from cv2 import cv2
import numpy as np
from numpy import ndarray


def find_template(im_source: ndarray, im_template: ndarray, threshold: float = 0.5,
                  auto_scale: tuple = None,
                  edge: bool = False,
                  debug: bool = False):
    """
    在im_source中查找im_template的匹配位置，返回最匹配的那个结果，内部调用find_all_template实现
    Args:
        im_source(string): 源图(大图)，opencv格式的图片
        im_template(string): 需要查找的图片(小图)，opencv格式的图片
        threshold: 阈值，当匹配度小于该阈值的时候，就忽略掉，是一个-1~1之间的值，通常小于0.5，匹配度就相当低了
        auto_scale: 是否自动缩放im_template来查找匹配，如果为None表示不缩放，如果需要缩放，那么传一个tuple：(min_scale, max_scale, step)，
        其中min_scale和max_scale分别是缩放倍数的下限和上限，都是小数，min_scale介于0~1之间，max_scale大于1,
        edge: 是否做边缘提取后再匹配，缺省为False，如果设置为True，会把源图和模板图，都基于Canny算法提取边缘，然后再做匹配
        debug: 是否不输出中间处理步骤和处理时间
    Returns:
        匹配结果对象,包含如下属性:
        result: 匹配区域的中心点
        rectangle: 匹配区域的四角坐标
        confidence: 匹配程度, 是一个-1~1之间的值, 约大表示匹配度越高
        如果没有找到符合条件的匹配结果, 返回None

    """
    result = find_all_template(im_source, im_template, threshold, 1, auto_scale, edge, debug)
    return result[0] if result else None


def find_all_template(im_source: ndarray, im_template: ndarray, threshold: float = 0.5, maxcnt: int = 0,
                      auto_scale=None,
                      edge: bool = False,
                      debug: bool = False):
    """
    在im_source中查找im_template的匹配位置，返回指定数量的匹配结果

    Args:
        im_source(string): 源图(大图)，opencv格式的图片
        im_template(string): 需要查找的图片(小图)，opencv格式的图片
        threshold: 阈值，当匹配度小于该阈值的时候，就忽略掉，是一个-1~1之间的值，通常小于0.5，匹配度就相当低了
        maxcnt: 最大匹配数量, 缺省为0, 即不限
        auto_scale: 是否自动缩放im_template来查找匹配，如果为None表示不缩放，如果需要缩放，那么传一个tuple：(min_scale, max_scale, step)，
        其中min_scale和max_scale分别是缩放倍数的下限和上限，都是小数，min_scale介于0~1之间，max_scale大于1,
        step是从min_scale开始，逐步尝试到max_scale之间的步长，缺省值为0.1，例如(0.8, 1.6, 0.2)
        edge: 是否做边缘提取后再匹配，缺省为False，如果设置为True，会把源图和模板图，都基于Canny算法提取边缘，然后再做匹配
        debug: 是否不输出中间处理步骤和处理时间
    Returns:
        匹配结果列表，每个结果包含以下属性：
        result: 匹配区域的中心点
        rectangle: 匹配区域的四角坐标
        confidence: 匹配程度, 是一个-1~1之间的值, 约大表示匹配度越高

    Raises:
        IOError: 读取文件失败
    """

    w, h = im_template.shape[1], im_template.shape[0]
    sw, sh = im_source.shape[1], im_source.shape[0]
    if w > sw or h > sh:
        raise RuntimeError(
            "source image size must larger than template image size, but not source is {}x{}, template is {}x{}",
            sw, sh, w, h)

    start_time = time.time()
    gray_template = _to_gray(im_template)
    gray_source = _to_gray(im_source)
    if debug:
        print("to_gray time: {}".format(time.time() - start_time))

    # 边界提取(来实现背景去除的功能)
    if edge:
        if debug:
            start_time = time.time()
        gray_template = cv2.Canny(gray_template, 100, 200)
        gray_source = cv2.Canny(gray_source, 100, 200)
        if debug:
            print("Canny time: {}".format(time.time() - start_time))

    result = _internal_find(gray_source, gray_template, maxcnt, threshold, debug)

    if len(result) == 0 and auto_scale is not None:
        scale_min = auto_scale[0]
        scale_max = auto_scale[1]
        step = auto_scale[2] if len(auto_scale)>2 else 0.1
        for scale in np.arange(scale_min, scale_max, step):
            resized = cv2.resize(gray_template, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_CUBIC)
            match_results = _internal_find(gray_source, resized, maxcnt, threshold, debug)
            if debug:
                print("try resize template in scale {} to find match".format(scale))
            if len(match_results) > 0:
                break

    return result


def _to_gray(image):
    channel = 1 if len(image.shape) == 2 else image.shape[2]
    if channel == 1:
        # if the image is gray, then keep it
        image_gray = image
    elif channel == 3:
        # if it's colorful, then convert it to gray
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channel == 4:
        # if it's colorful with transparent channel, then convert it to gray
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise RuntimeError('im_search have {} channel, which is unexpected!'.format(channel))
    return image_gray


def _internal_find(gray_source, gray_template, maxcnt, threshold, debug):
    start_time = time.time()

    w, h = gray_template.shape[1], gray_template.shape[0]
    sw, sh = gray_source.shape[1], gray_source.shape[0]

    if debug:
        start_time = time.time()
    res = cv2.matchTemplate(gray_source, gray_template, cv2.TM_CCOEFF_NORMED)
    if debug:
        print("matchTemplate time: {}".format(time.time() - start_time))
    if debug:
        start_time = time.time()
    result = []
    while True:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        if max_val < threshold:
            break

        left = top_left[0]
        top = top_left[1]
        middle_point = (left + w / 2, top + h / 2)
        result.append(dict(
            result=middle_point,
            rectangle=(top_left, (left, top + h), (left + w, top),
                       (left + w, top + h)),
            confidence=max_val
        ))
        if maxcnt and len(result) >= maxcnt:
            break
        # 用最小值填充当前结果的周边区域，避免下次找到重叠的结果
        x1 = left - w + 1 if left - w + 1 > 0 else 0
        x2 = left + w - 1 if left + w - 1 < sw else sw
        y1 = top - h + 1 if top - h + 1 > 0 else 0
        y2 = top + h - 1 if top + h - 1 < sh else sh
        res[y1:y2, x1:x2] = -1000

    if debug:
        print("find max time: {}".format(time.time() - start_time))

    return result
