# week 01 学习笔记
## 题目：
使用OpenCV利用颜色空间(lab，hsv，rgb，yuv等等都试一试)找出图中的红色和蓝色色块（分别）并标出色块中心的坐标。

## 解决方法：
使用cmake和make作为构建工具，但是由于之前使用opencv只是一些简单的应用，所以问了无敌的GPT解决的思路，由此有了大体的框架，剩下的便是学习函数。

## 过程：
在官方的文档中，文档的主要结构如下：
Main modules:
1. core. Core functionality
1. imgproc. Image Processing
1. imgcodecs. Image file reading and writing
1. videoio. Video I/O
1. highgui. High-level GUI
1. video. Video Analysis
1. calib3d. Camera Calibration and 3D Reconstruction
1. features2d. 2D Features Framework
1. objdetect. Object Detection
1. dnn. Deep Neural Network module
1. ml. Machine Learning
1. flann. Clustering and Search in Multi-Dimensional Spaces
1. photo. Computational Photography
1. stitching. Images stitching
1. gapi. Graph API

查看了主要的结构，imread返回的Mat以及Scalar，Scalar是（Template class for a 4-element vector derived from Vec. More. The type Scalar is widely used in OpenCV to pass pixel values）

接下来是一些函数的使用：
需要查找文档的有：
1. cv::inRange
1. cv::morphologyEx
1. cv::findContours
1. cv::moments

难点：
1. 使用什么样的方法，好在GPT给出了方法
1. 颜色的范围，使用了取色器确定了HSV的范围
1. 轮廓的选取