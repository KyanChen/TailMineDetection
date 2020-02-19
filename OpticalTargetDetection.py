# coding=utf-8
# 用于尾矿库检测
import os
import sys
import time
import tqdm
import datetime
import threading
import numpy as np
from scipy import misc
import tensorflow as tf
from bs4 import BeautifulSoup

from nets.ssd_vgg_512_Multi import SSDNet_vgg_512 as SSDNet
from utils import data_utils, gdal_utils, path_utils

import ogr
from utils.path_utils import get_filename
from utils.gdal_utils import get_geoTransform

# zoom为缩小的比例，1-overlapRatio为覆盖的比例
zoom = 1
overlapRatio = 0.7

class Block:
    """图像块结构体
    """
    def __init__(self, img_part, x, y):
        """初始化结构体
        """
        self.img_block = img_part
        self.mask_block = np.zeros((512*zoom, 512*zoom, 3), np.uint8)
        self.offset_x = x
        self.offset_y = y
        self.process_finished = False


class OpticalTargetDetection:
    """光学图像目标检测
    """
    def __init__(self, xml_path):
        """参数初始化
        """
        # 先验参数
        self.block_height = 512*zoom
        self.block_width = 512*zoom
        self.queue_maxsize = 150
        self.class_names = ['tail_mine']
        self.run_info = {'bndboxes': [], 'StartTime': '', 'EndTime': '', 'ProcessTime': '',
                         'ProduceTime': '', 'ReceiveTime': '', 'SolarAzimuth': '',
                         'SolarZenith': '', 'TopLeftLatitude': '', 'TopLeftLongitude': '',
                         'TopRightLatitude': '', 'TopRightLongitude': '', 'BottomRightLatitude': '',
                         'BottomRightLongitude': '', 'BottomLeftLatitude': '', 'BottomLeftLongitude': '',
                         'SatelliteID': '', 'AIModelVersion': '1.0', 'ManCorrect': 'False'}
        self.block_queue = []
        self.thread_lock = threading.Lock()
        self.read_finished = False

        # 输入参数
        self.model_path = ''
        self.threshold = 0.5
        self.inputFile_Path = ''
        self.outputFolder_Path = ''
        # self.logfile = './log'
        self.load_config(xml_path)
        # print(self.logfile)
        self.logfp = open(self.logfile, 'a+', encoding='utf-8')
        self.write_log('输入路径：%s' % self.inputFile_Path)
        self.write_log('输出路径：%s' % self.outputFolder_Path)

        # 图像参数
        self.height, self.width, _ = gdal_utils.get_image_shape(self.inputFile_Path)
        self.img_name = path_utils.get_filename(self.inputFile_Path, is_suffix=False)

    def load_config(self, xml_path):
        """读取xml配置文件
        """

        print(xml_path)
        root = BeautifulSoup(open(xml_path, encoding='utf-8'), 'lxml')

        root = root.findChild('interfacefile')
        file_root = root.findChild('fileroot')
        self.model_path = file_root.findChild('modelpath').text
        self.threshold = float(file_root.findChild('threshold').text)
        input_root = file_root.findChild('inputdir')
        self.inputFile_Path = input_root.findChild('imagefile').text
        self.outputFolder_Path = file_root.findChild('outputdir').text
        if not os.path.exists(self.outputFolder_Path):
            os.mkdir(self.outputFolder_Path)
        self.log_path='./log'
        # if not os.path.exists(self.log_path):
        #     os.makedirs(self.log_path)
        self.logfile = os.path.join(self.outputFolder_Path,
                                    '%s.txt' % (datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))


    def write_log(self, log_content):
        """写处理日志
        """
        self.logfp.write('%s\n' % str(log_content))
        self.logfp.flush()

    def multi_thread_read(self):
        """多线程读图
        """
        self.write_log('Reading %s' % self.img_name)
        img_read = gdal_utils.GDALReader(self.inputFile_Path)
        y = 0
        flag_x_skip = False
        flag_y_skip = False
        while y < self.height:
            if self.block_height + y > self.height:
                y = self.height - self.block_height
                flag_y_skip = True
            x = 0
            flag_x_skip = False
            while x < self.width:
                if x + self.block_width > self.width:
                    x = self.width - self.block_width
                    flag_x_skip = True
                img_part = img_read.read_image(x, y, self.block_width, self.block_height,
                                               data_format='NUMPY_FORMAT')

                while len(self.block_queue) >= self.queue_maxsize:
                    time.sleep(0.1)
                self.thread_lock.acquire()
                self.block_queue.append(Block(img_part, x, y))
                self.thread_lock.release()
                if flag_x_skip:
                    break
                x += int(512*zoom*overlapRatio)
            if flag_y_skip:
                break
            y += int(512*zoom*overlapRatio)
        self.read_finished = True
        self.write_log('Reading done')

    def multi_thread_write(self):
        """多线程写图
        """
        if not os.path.exists(self.outputFolder_Path):
            os.mkdir(self.outputFolder_Path)

        self.write_log('Writing %s' % self.img_name)
        # geoTranfsorm=gdal_utils.get_geoTransform(self.inputFile_Path)
        img_write = []
        for i in range(len(self.class_names)):
            save_path = os.path.join(self.outputFolder_Path, '%s_%s_MASK.tiff' % (self.img_name, self.class_names[i]))
            img_write.append(gdal_utils.GDALWriter(save_path, self.width, self.height, 1, 'uint8',
                                                   geo_transform=gdal_utils.get_geoTransform(self.inputFile_Path),projection=gdal_utils.get_projection(self.inputFile_Path)))

        while self.block_queue or not self.read_finished:
            if self.block_queue and self.block_queue[0].process_finished:
                self.thread_lock.acquire()
                block = self.block_queue.pop(0)
                self.thread_lock.release()
                for i in range(len(self.class_names)):
                    img_write[i].save_image(block.mask_block[:, :, i], block.offset_x, block.offset_y, data_format='NUMPY_FORMAT')
            else:
                time.sleep(0.1)
        self.write_log('Writing done')

    def draw_mask(self, localizations, scores, labels, width_offset, height_offset, threshold=0.5):
        index = np.nonzero(np.logical_and(scores > threshold, labels > 0))
        localizations = localizations * 512
        localizations = np.int32(localizations)
        mask = np.zeros((512*zoom, 512*zoom, 3), np.uint8)
        index = np.array(index)
        index = np.reshape(index, [-1])
        count = 0
        loc_info = []
        for i in range(len(index)):
            idx = index[i]
            loc = localizations[idx, :] * zoom
            count += 1
            label = labels[idx]
            mask[loc[0]:loc[2], loc[1]:loc[3], label - 1] = 255
            loc_info.append(
                [self.class_names[label - 1], loc[0] + height_offset, loc[1] + width_offset, loc[2] + height_offset,
                 loc[3] + width_offset])
        return count, mask, loc_info

    def multi_thread_interface(self):
        """多线程处理过程
        """
        self.write_log('building networks...')
        ssd_nets = SSDNet()
        ssd_anchors = ssd_nets.place_anchors()
        test_img_placeholder = tf.placeholder(tf.float32, [1, None, None, 3])
        test_img = data_utils.preprocess_test(test_img_placeholder)
        test_op = ssd_nets.test_op(test_img, ssd_anchors, reuse=False)
        self.write_log('Initial parameters...')
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self.write_log('Loading Model...')
        if 'checkpoint' in os.listdir(self.model_path):
            print('restore from last model....')
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('restore ' + ckpt.model_checkpoint_path)

        self.write_log('Processing %s' % self.img_name)

        pbar = tqdm.tqdm(total=self.height * self.width // (self.block_height * self.block_width))
        totoal_count=self.height * self.width // (self.block_height * self.block_width)
        start=datetime.datetime.now()
        sub_start = datetime.datetime.now()
        while self.block_queue or not self.read_finished:
            for block in self.block_queue:
                if not block.process_finished:

                    img_part = gdal_utils.swap_band(block.img_block)
                    # add
                    img_part = misc.imresize(img_part, [512, 512])
                    img_part = np.expand_dims(img_part, 0)
                    img_part = img_part.astype(np.float32) / 255.
                    sub_end=datetime.datetime.now()

                    sub_start=datetime.datetime.now()
                    [_, localization, scores, labels, index, _] = sess.run(test_op,
                                                                           feed_dict={test_img_placeholder: img_part})
                    sub_end = datetime.datetime.now()
                    sub_start = datetime.datetime.now()
                    count, mask, loc = self.draw_mask(localization[index, :], scores[index],
                                                      labels[index], block.offset_x, block.offset_y, threshold=self.threshold, )
                    block.mask_block = misc.imresize(mask, [self.block_height, self.block_width], interp='nearest')

                    block.process_finished = True
                    self.run_info['bndboxes'].extend(loc)
                    pbar.update(1)
                    break

        self.write_log('Process done')
        pbar.close()
        end=datetime.datetime.now()
        print((end-start).seconds)

    def run(self):
        """运行程序
        """
        begin = datetime.datetime.now()
        self.run_info['StartTime'] = begin.strftime('%Y:%m:%d:%H:%M:%S')

        thread_read = threading.Thread(target=self.multi_thread_read)
        thread_read.start()
        thread_write = threading.Thread(target=self.multi_thread_write)
        thread_write.start()

        try:
            self.multi_thread_interface()
        except Exception as e:
            print('程序出错', e)
            self.write_log('Except occur %s' % e)
            self.run_info['IsSuccess'] = 'False'
            self.run_info['IsSkip'] = 'True'
            self.run_info['FailInfo'] = e
            self.run_info['SkipInfo'] = e
        else:
            self.run_info['IsSuccess'] = 'True'
            self.run_info['IsSkip'] = 'False'
            self.run_info['FailInfo'] = ''
            self.run_info['SkipInfo'] = ''

        end = datetime.datetime.now()
        self.run_info['EndTime'] = end.strftime('%Y:%m:%d:%H:%M:%S')
        self.run_info['ProcessTime'] = (end - begin).seconds

        self.write_xml()
        self.write_shp(0)
        self.logfp.close()
        print(self.run_info['ProcessTime'])

    def write_shp(self,label):
        """
        --convert the pixel location to latitude and longitude
        --write all objects in a certain image into a single shp file
        """
        shapefilename = os.path.join(self.outputFolder_Path, get_filename(self.inputFile_Path,is_suffix=False)+'_%s_shp.shp'%self.class_names[label])
        class_bndbox = self.run_info['bndboxes']

        # 获取图像六元数 0-经度 1-经度分辨率 3-纬度 5-纬度分辨率
        lat_lon_init = get_geoTransform(self.inputFile_Path)

        # 注册所有的驱动
        ogr.RegisterAll()

        # 创建数据
        strDriverName = 'ESRI Shapefile'
        oDriver = ogr.GetDriverByName(strDriverName)
        if oDriver == None:
            print("%s 驱动不可用！\n", strDriverName)

        # 创建数据源
        oDS = oDriver.CreateDataSource(shapefilename)
        if oDS == None:
            print("创建文件【%s】失败！", shapefilename)

        # 创建图层
        outLayer = oDS.CreateLayer('detection', geom_type=ogr.wkbPolygon)
        # papszLCO = []
        # outLayer = oDS.CreateLayer("TestPolygon", None, ogr.wkbPolygon, papszLCO)

        # oDefn = outLayer.GetLayerDefn()
        # # 创建矩形要素
        # oFeatureRectangle = ogr.Feature(oDefn)
        # oFeatureRectangle.SetField(0, 1)
        # oFeatureRectangle.SetField(1, "rect1")
        # geomRectangle = ogr.CreateGeometryFromWkt("POLYGON ((30 0,60 0,60 30,30 30,30 0))")
        # oFeatureRectangle.SetGeometry(geomRectangle)
        # outLayer.CreateFeature(oFeatureRectangle)


        # create fields


        fieldDefn2 = ogr.FieldDefn('class', ogr.OFTInteger)
        fieldDefn2.SetWidth(10)
        outLayer.CreateField(fieldDefn2, 1)


        # get feature defintion
        outFeatureDefn = outLayer.GetLayerDefn()



        for object in class_bndbox:

            # wkt = "POINT(%f %f)" % (float(pointListX[i]), float(pointListY[i]))
            # point = ogr.CreateGeometryFromWkt(wkt)
            # outFeature.SetGeometry(point)


            if self.class_names.index(object[0])!=label:
                continue
            # 坐标转换
            Ymin = object[1]*lat_lon_init[5]+lat_lon_init[3]
            Xmin = object[2]*lat_lon_init[1]+lat_lon_init[0]
            Ymax = object[3]*lat_lon_init[5]+lat_lon_init[3]
            Xmax = object[4]*lat_lon_init[1]+lat_lon_init[0]


            oFeatureRectancle=ogr.Feature(outFeatureDefn)
            oFeatureRectancle.SetField(0,self.class_names.index(object[0])+1)
            polygon_cmd='POLYGON ((%f %f,%f %f,%f %f,%f %f,%f %f))'%(Xmin,Ymin,Xmin,Ymax,Xmax,Ymax,Xmax,Ymin,Xmin,Ymin)
            geomRectancle=ogr.CreateGeometryFromWkt(polygon_cmd)
            oFeatureRectancle.SetGeometry(geomRectancle)

            outLayer.CreateFeature(oFeatureRectancle)
            oFeatureRectancle.Destroy()

        oDS.Destroy()

        print('shp finished!')
        pass

    def write_xml(self):
        self.run_info['SatelliteID'] = self.img_name.split('_')[0]
        # 写入每个类别的xml
        for i in range(len(self.class_names)):
            one_loc_info = self.run_info['bndboxes']
            self.write_one_result(one_loc_info, self.img_name, self.class_names[i])

    def write_one_result(self, result_loc, img_name, class_name):
        """一个简单记录的文件，一个记录所有的框
        """
        count = 0
        xml_path = os.path.join(self.outputFolder_Path, '%s_%s_bndboxes.xml' % (img_name, class_name))
        soup = BeautifulSoup(features='lxml')
        object_tag = soup.new_tag('object')
        for loc in result_loc:
            if loc[0] != class_name:
                continue
            bnd_tag = soup.new_tag('bndbox')
            tag = soup.new_tag('class')
            tag.insert(0, loc[0])
            bnd_tag.append(tag)
            tag = soup.new_tag('ymin')
            tag.insert(0, str(loc[1]))
            bnd_tag.append(tag)
            tag = soup.new_tag('xmin')
            tag.insert(0, str(loc[2]))
            bnd_tag.append(tag)
            tag = soup.new_tag('ymax')
            tag.insert(0, str(loc[3]))
            bnd_tag.append(tag)
            tag = soup.new_tag('xmax')
            tag.insert(0, str(loc[4]))
            bnd_tag.append(tag)
            object_tag.append(bnd_tag)
            count += 1
        soup.append(object_tag)
        fp = open(xml_path, 'w')
        fp.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        fp.write(soup.prettify())
        fp.close()

        xml_path = os.path.join(self.outputFolder_Path, '%s_%s.xml' % (img_name, class_name))
        soup = BeautifulSoup(features='lxml')
        root = BeautifulSoup.new_tag(soup, 'AIProductFile')
        soup.append(root)

        fileHeader_tag = soup.new_tag('FileHeader')
        tag = soup.new_tag('type')
        tag.insert(0, class_name)
        fileHeader_tag.append(tag)
        tag = soup.new_tag('IsSkip')
        tag.insert(0, self.run_info['IsSkip'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('SkipInfo')
        tag.insert(0, self.run_info['SkipInfo'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('IsSuccess')
        tag.insert(0, self.run_info['IsSuccess'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('FailInfo')
        tag.insert(0, self.run_info['FailInfo'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('ManCorrect')
        tag.insert(0, self.run_info['ManCorrect'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('StartTime')
        tag.insert(0, self.run_info['StartTime'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('EndTime')
        tag.insert(0, self.run_info['EndTime'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('ProcessTime')
        tag.insert(0, str(self.run_info['ProcessTime']))
        fileHeader_tag.append(tag)
        root.append(fileHeader_tag)

        fileBody_tag = soup.new_tag('FileBody')
        metaInfo_tag = soup.new_tag('MetaInfo')
        tag = soup.new_tag('ProduceTime')
        tag.insert(0, self.run_info['ProduceTime'])
        metaInfo_tag.append(tag)
        metaInfo_tag = soup.new_tag('MetaInfo')
        tag = soup.new_tag('ReceiveTime')
        tag.insert(0, self.run_info['ReceiveTime'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('ReceiveTime')
        tag.insert(0, self.run_info['ReceiveTime'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('SolarAzimuth')
        tag.insert(0, self.run_info['SolarAzimuth'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('SolarZenith')
        tag.insert(0, self.run_info['SolarZenith'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('TopLeftLatitude')
        tag.insert(0, self.run_info['TopLeftLatitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('TopLeftLongitude')
        tag.insert(0, self.run_info['TopLeftLongitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('TopRightLatitude')
        tag.insert(0, self.run_info['TopRightLatitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('TopRightLongitude')
        tag.insert(0, self.run_info['TopRightLongitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('BottomRightLatitude')
        tag.insert(0, self.run_info['BottomRightLatitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('BottomRightLongitude')
        tag.insert(0, self.run_info['BottomRightLongitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('BottomLeftLatitude')
        tag.insert(0, self.run_info['BottomLeftLatitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('BottomLeftLongitude')
        tag.insert(0, self.run_info['BottomLeftLongitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('SatelliteID')
        tag.insert(0, self.run_info['SatelliteID'])
        metaInfo_tag.append(tag)
        fileBody_tag.append(metaInfo_tag)

        RSAIInfo_tag = soup.new_tag('RSAIInfo')
        tag = soup.new_tag('AIModelVersion')
        tag.insert(0, self.run_info['AIModelVersion'])
        RSAIInfo_tag.append(tag)
        tag = soup.new_tag('ObjectCount')
        tag.insert(0, str(count))
        RSAIInfo_tag.append(tag)
        fileBody_tag.append(RSAIInfo_tag)

        root.append(fileBody_tag)

        fp = open(xml_path, 'w')
        fp.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        fp.write(soup.prettify())
        fp.close()


def main():
    # print(sys.argv[1])
    # optical = OpticalTargetDetection(sys.argv[1])
    optical = OpticalTargetDetection('./xml/input.xml')
    optical.run()


if __name__ == "__main__":
    main()
