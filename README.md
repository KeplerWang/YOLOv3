# Minimal PyTorch implementation of YOLOv3 

## Overview
After reading the YOLO papers,
one idea that implementing this object detection algorithm comes out. 
I read a lot of tutorials about this topic, such as build your own YOLO from scratch.
But one thing I found out is that there's few repositories friendly to people who want to implement it but not use it directly.

So, here comes mine. 
This repository is created in order to share my understanding of YOLO(v1, v3, and some other series).

But sorry to note that at the begining I don't offer any training results on public datasets like PASCAL_VOC2007 and COCO, 
just because I don't have enough time and adequate GPUs.
If you find this repository useful and have tried it on certain datasets, 
I would appreciate it that you offer the results.

## Todo list

- [ ] Train on PASCAL_VOC2007 and supplement the results.

- [ ] Add some training tricks, such as train backbone first?

## Reference
- Paper [YOLOv1](https://arxiv.org/abs/1506.02640) [YOLOv3](https://arxiv.org/abs/1804.02767)
- How to implement a YOLO (v3) object detector from scratch in PyTorch? [Github](https://github.com/ayooshkathuria/pytorch-yolo-v3) [Blog](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
- YOLOv3 from Scratch [Github](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3) [Video](https://www.youtube.com/watch?v=Grir6TZbc1M)
- Pytorch 搭建自己的YOLO3目标检测平台 [Github](https://github.com/bubbliiiing/yolo3-pytorch) [Blog](https://blog.csdn.net/weixin_44791964/article/details/105310627) [Video](https://www.bilibili.com/video/BV1Hp4y1y788?from=search&seid=18024492462159540693&spm_id_from=333.337.0.0)
- mAP calculation [Github](https://github.com/Cartucho/mAP)
