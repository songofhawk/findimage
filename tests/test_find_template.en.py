from cv2 import cv2
import time

from findimage import find_template

image_origin = cv2.imread('amazon_menu.png')
image_template = cv2.imread('amazon_arrow.png')

start_time = time.time()
match_result = find_template(image_origin, image_template)
print("total time: {}".format(time.time() - start_time))

img_result = image_origin.copy()
rect = match_result['rectangle']
cv2.rectangle(img_result, (rect[0][0], rect[0][1]), (rect[3][0], rect[3][1]), (0, 0, 220), 2)
cv2.imwrite('../image/find_template_result.en.png', img_result)
