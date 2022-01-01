from cv2 import cv2
import time

from findimage import find_all_template

image_origin = cv2.imread('seg_course_whole_page.png')
image_template = cv2.imread('seg_sharp.png')

start_time = time.time()
match_results = find_all_template(image_origin, image_template, 0.5, 50, debug=True)
print("total time: {}".format(time.time() - start_time))

img_result = image_origin.copy()
for match_result in match_results:
    rect = match_result['rectangle']
    cv2.rectangle(img_result, (rect[0][0], rect[0][1]), (rect[3][0], rect[3][1]), (0, 0, 220), 2)
cv2.imwrite('result.png', img_result)
