[English version](README.en.md)

# findimage - 在大图中找小图
本项目不是图像搜索，不能从一堆图片中找到相似的那张，而是从一张大图中定位给定模板小图的位置。

find the template image (smaller) in a source image (bigger)

以前这种功能，我都是用aircv实现的，但网易这个项目很久没有维护了，提交PR也没人处理，所以单独做了一个。
原项目还有一些别的图像处理API，本项目专注于找小图功能，所以干脆就叫findimage。

和原aircv相比，findimage有以下这些改进:
* 支持直接传入灰度图(虽然函数内调用opencv的时候,都是使用灰度图完成的,原aircv项目却要求传入的图片必须包含bgr三个通道,不然会报错)
* 支持背景透明的图片
* 优化了find_all_template方法的性能，用numpy的切片赋值代替floodFill方法来避免重叠，大概会缩短1/4的总体查找时间

To find a template image(smaller) in a source image(bigger)

This project is inspired from https://github.com/NetEaseGame/aircv.git, which is not maintained for a long time.

There are several improvements and changes in this projects:
* support finding grayscale image, either source or template
* support finding image with transparent channel
* optimized the performance of find_all, use numpy slicing set data instead of floodFill
* removed methods that are not related to finding images

## 安装
```shell
pip install findimage
```

## 使用示例1
比如我们对“思否”课程菜单截图如下：
![思否课程菜单-标准](https://github.com/songofhawk/findimage/raw/main/image/seg_course_menu.png)

我们想从中找到#的位置，可以提供一张小模板图：
![思否课程菜单-标准](https://github.com/songofhawk/findimage/raw/main/image/seg_sharp.png)

然后调用find_template方法：

```python
from cv2 import cv2
from findimage import find_template

image_origin = cv2.imread('seg_course_whole_page.png')
image_template = cv2.imread('seg_sharp.png')

match_result = find_template(image_origin, image_template)
```

得到的match_result，标识了第一个#在源图中的中心点位置，矩形区域四角坐标 和 匹配度。

```json
{
    "result": (x,y),        #tuple，表示识别结果的中心点
    "rectangle":[            #二位数组，表示识别结果的矩形四个角
        [left, top],
        [left, bottom],
        [right, top],
        [right, bottom]
    ],
    "confidence": percentage   #识别结果的匹配度,在-1~1之间，越大匹配度越高, 如果为1，表示按像素严格匹配
}
```

我们可以用这个结果，在源图上标识出匹配的位置：
```python
img_result = image_origin.copy()
rect = match_result['rectangle']
cv2.rectangle(img_result, (rect[0][0], rect[0][1]), (rect[3][0], rect[3][1]), (0, 0, 220), 2)
cv2.imwrite('find_all_template_result.en.png', img_result)
```

结果如下图所示：
![find_template匹配结果](https://github.com/songofhawk/findimage/raw/main/image/find_template_result.png)

## 使用示例2——指定匹配度
find_template方法有一个threshold参数，如果设置了这个值，那么只有大于指定匹配度的图像，才能被查找出来：
```python
match_result = find_template(image_origin, image_template, 0.8)
```
这个参数的取值范围是0~1，缺省值是0.5，这个值设置得越低，越容易找到结果，但也越容易找错；设置得越高，结果匹配越准确，但也可能找不到结果

## 使用示例3——查找所有结果
一张大图上不一定只有一个小图匹配结果，也可能有多个，如果需要返回多个结果，可以使用find_all_template方法:
```python
from cv2 import cv2
import time

from findimage import find_all_template

image_origin = cv2.imread('seg_course_menu.png')
image_template = cv2.imread('seg_sharp.png')

start_time = time.time()
# 查找所有匹配
match_results = find_all_template(image_origin, image_template, 0.8, 50)
print("total time: {}".format(time.time() - start_time))

# 绘制结果图
img_result = image_origin.copy()
for match_result in match_results:
    rect = match_result['rectangle']
    cv2.rectangle(img_result, (rect[0][0], rect[0][1]), (rect[3][0], rect[3][1]), (0, 0, 220), 2)
cv2.imwrite('find_all_template_result.en.png', img_result)
```
find_all_template方法，提供一个额外的maxcnt参数，用于限制最多查找多少个结果，缺省为0（即不限），以上代码会把所有结果绘制出来：

![find_all_template匹配结果](https://github.com/songofhawk/findimage/raw/main/image/find_all_template_result.png)
