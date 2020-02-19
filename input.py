from threading import Thread
from queue import  Queue
import path_utils
import gdal_utils
import os
import glob
import numpy as np
import pandas as pd
import time

IMG_SIZE = 512

def cal_result(localizations, scores, labels, offsets,threshold=0.5):
    """
    :param localizations: 目标的位置坐标（x1，y1）和（x2,y2），我还没验证，相对关系待验证
    :param scores: 目标对应的得分
    :param labels: 目标类别
    :param offsets: 切片左上角位置
    :param threshold: 阈值
    :return: 
    """
    loc_info = []
    score_info = []
    label_info = []
    for localization, score, label, offset in zip(localizations, scores, labels, offsets):
        index = np.logical_and(score>threshold,label>0)
        localization = localization[index]
        score = score[index]
        label = label[index]
        flags = [True,False,True,False]
        localization = localization*512
        localization = np.int64(localization)
        localization = np.where(flags,localization+offset[0],localization+offset[1])
        loc_info.append(localization)
        score_info.append(score)
        label_info.append(label)

    return loc_info, score_info, label_info


class DataInputTest():
    """
    多切片一起检测
    """
    def __init__(self, batch_size, img_file):
        self.img_file=img_file
        self.bacth_size = batch_size
        self.offsets = self.get_loc(self.img_file)
        self.epoch_size = len(self.offsets)//self.bacth_size
        if self.epoch_size*self.bacth_size<len(self.offsets):
            self.epoch_size+=1

        self.iter = 0


    def get_loc(self, img_file):
        offsets = []
        height, width, _ = gdal_utils.get_image_shape(img_file)
        for x in range(0, height, IMG_SIZE):
            x = height - IMG_SIZE if IMG_SIZE + x > height else x
            for y in range(0, width, IMG_SIZE):
                y = width - IMG_SIZE if y + IMG_SIZE > width else y
                offsets.append([x,y])

        return offsets

    def __iter__(self):
        return self

    def __next__(self):
        start = time.time()
        if self.iter == self.epoch_size:
            raise StopIteration

        offsets = self.offsets[self.iter*self.bacth_size:min((self.iter+1)*self.bacth_size,len(self.offsets))]
        self.iter+=1
        imgs = []
        for offset in offsets:
            patch_source = gdal_utils.read_image(self.img_file, offset[1], offset[0], IMG_SIZE, IMG_SIZE,
                                                    data_format='NUMPY_FORMAT')
            patch = gdal_utils.swap_band(patch_source)
            patch = patch.astype(np.float32) / 255.
            imgs.append(patch)
        end = time.time()
        #print("read time:%d"%(end-start))

        return self.iter, (imgs, offsets)





