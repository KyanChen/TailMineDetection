'''李汶原 20180804
gdal读取图像的辅助工具库
'''
import gdal
import numpy as np
import os


def get_image_shape(img_path):
    '''
    获取图像的尺寸，格式为(height，width，bands)
    :param img_path: 
    :return: 
    '''

    dataset = gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开" % img_path)
        exit(-1)
    im_width = dataset.RasterXSize  # 图像的列数
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount

    return im_height,im_width,im_bands

#待补充
# def save_image(img_path,img,width_offset,height_offset,width,height,
#                geoTranfsorm=None,proj=None,data_format='GDAL_FORMAT'):
#     '''
#     保存图像
#     :param img_path: 保存的路径
#     :param img:
#     :param geoTranfsorm:
#     :param proj:
#     :return:
#     '''
#     if data_format not in ['GDAL_FORMAT','NUMPY_FORMAT']:
#         raise Exception('data_format参数错误')
#     if 'uint8' in img.dtype.name:
#         datatype=gdal.GDT_Byte
#     elif 'int16' in img.dtype.name:
#         datatype=gdal.GDT_CInt16
#     else:
#         datatype=gdal.GDT_Float32
#     if len(img.shape)==3:
#         if data_format=='NUMPY_FORMAT':
#             img = np.swapaxes(img, 1, 2)
#             img = np.swapaxes(img, 0, 1)
#         im_bands,im_height,im_width=img.shape
#     elif len(img.shape)==2:
#         img=np.array([img])
#         im_bands,im_height, im_width = img.shape
#     else:
#         im_bands,(im_height,im_width)=1,img.shape
#
#     driver=gdal.GetDriverByName("GTIFF")
#     # if os.path.exists(img_path):
#     #     dataset=driver.Create(img_path,im_width,im_height,im_bands,datatype)
#     dataset = gdal.Open(img_path)
#     if dataset is None:
#         print("文件%s无法打开" % img_path)
#         exit(-1)
#     full_height,full_width,_=get_image_shape(img_path)
#
#     if width_offset+width>full_width:
#         block_width=full_width-width_offset
#     if height_offset+height>full_height:
#         block_height=full_height-height_offset
#     if geoTranfsorm:
#         dataset.SetGeoTransform(geoTranfsorm)
#     if proj:
#         dataset.SetProjection(proj)
#     for i in range(im_bands):
#         dataset.GetRasterBand(i+1).WriteArray(img[i],width_offset,height_offset)

def save_full_image(img_path,img,geoTranfsorm=None,proj=None,data_format='GDAL_FORMAT'):
    '''
    保存图像
    :param img_path: 保存的路径
    :param img: 
    :param geoTranfsorm: 
    :param proj: 
    :return: 
    '''
    if data_format not in ['GDAL_FORMAT','NUMPY_FORMAT']:
        raise Exception('data_format参数错误')
    if 'uint8' in img.dtype.name:
        datatype=gdal.GDT_Byte
    elif 'uint16' in img.dtype.name:
        datatype=gdal.GDT_UInt16
    else:
        datatype=gdal.GDT_Float32
    if len(img.shape)==3:
        if data_format=='NUMPY_FORMAT':
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 1)
        im_bands,im_height,im_width=img.shape
    elif len(img.shape)==2:
        img=np.array([img])
        im_bands,im_height, im_width = img.shape
    else:
        im_bands,(im_height,im_width)=1,img.shape

    driver=gdal.GetDriverByName("GTIFF")
    dataset=driver.Create(img_path,im_width,im_height,im_bands,datatype)
    if geoTranfsorm:
        dataset.SetGeoTransform(geoTranfsorm)
    if proj:
        dataset.SetProjection(proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(img[i])

def read_full_image(img_path,scale_factor=1,as_rgb=True,as_8bit=True,
               data_format='GDAL_FORMAT'):
    '''
    一次读取整张图片
    :param img_path: 
    :param scale_factor: 
    :param as_rgb: 
    :param data_format: 
    :return: 
    '''
    im_height, im_width, _ = get_image_shape(img_path)
    img = read_image(img_path, 0,0, im_width,im_height, scale_factor, as_rgb, data_format, as_8bit=as_8bit)
    return img

def read_image(img_path,width_offset,height_offset,width,height,scale_factor=1,as_rgb=True,
               data_format='GDAL_FORMAT',as_8bit=True):
    '''
    读取图片,支持分块读取,若读取的尺寸超过图像的实际尺寸，则在边界补0
    :param img_path: 要读取的图片的路径
    :param width_offset: x方向的偏移量
    :param height_offset: y方向上的偏移量
    :param width: 要读取的图像块的宽度
    :param height: 要读取的图像块的高度
    :param scale_factor:缩放比例
    :param as_rgb:是否将灰度图转化为rgb图
    :param data_format:返回结果的格式,有两种取值：'GDAL_FORMAT','NUMPY_FORMAT'
                    'GDAL_FORMAT':返回图像的shape为'(bands,height,width)'
                    'NUMPY_FORMAT':返回图像的尺寸为(height,width,bands)
                    每种格式下返回的图像的shape长度都为3
    :return: 
    '''
    if data_format not in ['GDAL_FORMAT','NUMPY_FORMAT']:
        raise Exception('data_format参数错误')
    dataset=gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开"%img_path)
        exit(-1)

    im_width = dataset.RasterXSize  # 图像的列数
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    scale_width = int(width / scale_factor)
    scale_height = int(height / scale_factor)
    #判断索引是否越界，只读取不越界部分的图像，其余部分补0
    block_width=width
    block_height=height
    if width_offset+width>im_width:
        block_width=im_width-width_offset
    if height_offset+height>im_height:
        block_height=im_height-height_offset
    scale_block_width = int(block_width / scale_factor)
    scale_block_height = int(block_height / scale_factor)
    buf_obj=np.zeros((im_bands,scale_block_height,scale_block_width),dtype=np.uint16)
    im_data=dataset.ReadAsArray(width_offset,height_offset,block_width,block_height,
                                buf_obj,scale_block_width,scale_block_height)

    if im_data.dtype=='uint16' and np.max(im_data)>255 and as_8bit==True:
        im_data=im_data/4.
    elif im_data.dtype=='float32':
        raise '不支持float32类型'
    if im_data.dtype=='uint16' and as_8bit==False:
        im_data=np.array(im_data, np.uint16)  #此时的shape为(bands,height,width)?待验证height和width的顺序
    else:
        im_data = np.array(im_data, np.uint8)
    if width!=block_width or height!=block_height:
        im_data=np.pad(im_data,((0,0),(0,scale_height-scale_block_height),(0,scale_width-scale_block_width)),mode='constant')

    if im_bands==1 and as_rgb:
        im_data = np.tile(im_data,(3,1,1))
    elif im_bands==4 and as_rgb:
        im_data = im_data[0:-1,:,:]

    if data_format=='NUMPY_FORMAT':
        im_data=np.swapaxes(im_data,0,1)
        im_data=np.swapaxes(im_data,1,2)

    return im_data

def get_geoTransform(img_path):
    dataset = gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开" % img_path)
        exit(-1)
    geotransform=dataset.GetGeoTransform()
    return geotransform

def get_projection(img_path):
    dataset = gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开" % img_path)
        exit(-1)
    projection=dataset.GetProjection()
    return projection


def swap_band(img):
    result_img = np.zeros_like(img)
    result_img[ :, :,0] = img[:, :,2]
    result_img[:, :,2] = img[:, :,0]
    result_img[ :, :,1] = img[ :, :,1]

    if img.shape[-1]==4:
        result_img[:,:,-1] = img[:,:,-1]
    return result_img