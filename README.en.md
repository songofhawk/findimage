[中文版](README.md)

# findimage

**To Find a smaller image in a larger image, in another word, find a template image in a source image**

*This project is inspired from project [aircv](https://github.com/NetEaseGame/aircv.git), which is not maintained for a long time.*

There are several improvements and changes:
* support finding grayscale image, either source or template
* support finding image with transparent channel
* optimized the performance of find_all, use numpy slicing set data instead of floodFill
* removed methods that are not related to finding images

*The API of this project is compatible with aircv* 

## Installation
```
pip install findimage
```

## Demo1
for example, we got a snapshot from Amazon website:

![amazon menu standard](https://github.com/songofhawk/findimage/raw/main/image/amazon_menu.png)

We'd liket to find a '>' from the source image, then we can offer a little template image:

![amazon menu arrow](https://github.com/songofhawk/findimage/raw/main/image/amazon_arrow.png)

then call `find_template` method:

```
from cv2 import cv2
from findimage import find_template

image_origin = cv2.imread('seg_course_whole_page.png')
image_template = cv2.imread('seg_sharp.png')

match_result = find_template(image_origin, image_template)
```

get a matched return result, which indicate where is the first arrow icon in the source image, 
including coordinates of center and 4 corner points, and match confidence.
```
{
    "result": (x,y),        # tuple，indicates the center coordinates
    "rectangle":[            # 2 dimentional array, indicats 4 corners
        [left, top],
        [left, bottom],
        [right, top],
        [right, bottom]
    ],
    "confidence": 0.8   # confidence of matching, a number between -1 and 1, more greater means more matchable. 
     # Means exactly match pixel by pixel if it's 1
}
```

We can use this result, to draw a label on the source image:
```
img_result = image_origin.copy()
rect = match_result['rectangle']
cv2.rectangle(img_result, (rect[0][0], rect[0][1]), (rect[3][0], rect[3][1]), (0, 0, 220), 2)
cv2.imwrite('result.png', img_result)
```

which looks like the following one:

![find_template match result](https://github.com/songofhawk/findimage/raw/main/image/find_template_result.en.png)

## Demo2: specifying the confidence
There's a threshold parameter in find_template function, if we specify it, 
then only the match confidence greater than it, will be returned.

```
match_result = find_template(image_origin, image_template, 0.8)
```

The range of this parameter is [0, 1], default is 0.5. 
The smaller it is, the easier to find results, but more mistakes may be given.
On the contrast, the greater, the more accurate, but maybe no result.

## Demo3: find all results
Certainly, we can find all results from the source image, by calling `find_all_template` method 

```
from cv2 import cv2
import time

from findimage import find_all_template

image_origin = cv2.imread('seg_course_menu.png')
image_template = cv2.imread('seg_sharp.png')

start_time = time.time()
# find all metched results
match_results = find_all_template(image_origin, image_template, 0.8, 50)
print("total time: {}".format(time.time() - start_time))

# draw results
img_result = image_origin.copy()
for match_result in match_results:
    rect = match_result['rectangle']
    cv2.rectangle(img_result, (rect[0][0], rect[0][1]), (rect[3][0], rect[3][1]), (0, 0, 220), 2)
cv2.imwrite('find_all_template_result.en.png', img_result)
```
There's an extra parameter named `maxcnt`, to limit max match result, default is 0(means no limitation).

![find_all_template匹配结果](https://github.com/songofhawk/findimage/raw/main/image/find_all_template_result.en.png)
