# GraduationPro_ImageTransfer
DLUT本科毕业设计——基于深度学习的艺术风格转化应用（Art style transfer application based on deep learning）

## 主要参考论文列表（Paper References）：
1.《Image Style Transfer Using Convolutional Neural Networks》

2.《A Neural Algorithm of Artistic Style》

3.《Deep Photo Style Transfer》

4.《Texture Networks，Feed-forward Synthesis of Textures and Stylized Images》

5.《Perceptual Losses for Real-Time Style Transfer and Super-Resolution》

6.《Fast Patch-based Style Transfer of Arbitrary Style》

7.《Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization》

8.《Instance Normalization: The Missing Ingredient for Fast Stylization》

9.《Fully convolutional networks for semantic segmentation》

10.《A Closed-Form Solution to Natural Image Matting》

## 算法简单介绍
共实现四个算法完成风格转化任务，我将其分为两个大类：（1）艺术绘画风格转化算法（2）真实图像风格转化算法。每类包含两个算法，为快速算法和慢速算法。慢速艺术绘画风格转化算法主要参考《Image Style Transfer Using Convolutional Neural Networks》实现，其风格图像主要针对艺术绘画作品，而且训练一次产生一个风格化的图像；快速艺术绘画风格转化算法主要参考《Perceptual Losses for Real-Time Style Transfer and Super-Resolution》实现，其风格图像主要针对艺术绘画作品，每个风格图像训练一个模型，模型可快速对图像进行风格转化。慢速真实图像风格转化算法主要参考《Deep Photo Style Transfer》实现，其风格图像主要针对真实照片，训练一次产生一个风格化的图像，同时参考《Fully convolutional networks for semantic segmentation》的图像分割模型，计算图像的掩模；快速真实图像风格转化算法主要参考之前的三个算法自己简单实现的，训练数据集是爬虫爬取百度图库的图片，由于每次都要计算新的matting矩阵，因此算法训练时间很长，算法效果也有待提高。
## 网站简单介绍
Django框架实现简单网站提供风格转化服务。两个快速算法用于在线风格转化模块，两个慢速算法用于自定义风格转化模块。

##  实现过程中参考了GitHub上很多大牛的工作，在此一并表示感谢（Thanks）
lengstrom  project:https://github.com/lengstrom/fast-style-transfer

LouieYang  project:https://github.com/LouieYang/deep-photo-styletransfer-tf

DeepSegment  project:https://github.com/DeepSegment/FCN-GoogLeNet
