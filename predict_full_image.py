import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from scipy import misc
import gdal_utils
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import shutil

from config import FLAGS
from nets.ssd_vgg_512_Multi import SSDNet_vgg_512 as SSDNet
import utils.utils as utils
import dataset.dataSet_utils as data_utils

slim = tf.contrib.slim

channels = 3
OutPath = r'dataset\\negativeSamples20191201_2731'
if not os.path.exists(OutPath):
	os.makedirs(OutPath)

InPath = r'G:\ObjectDetection\ObjectDetectionData\资源中心尾矿库数据集_20191127\neg'
data_format = 'GF1_PMS2_E113.8_N26.8_20190924_L1A0004262731_FUSION_GEO.tiff'
img_files = glob.glob(os.path.join(InPath, data_format))



scale_factor = None
if '_FUSION_GEO' in data_format:
	scale_factor = 255.
elif '_FUSION_16Bit_GEO' in data_format:
	scale_factor = 1023.



def draw_boxes(img, localizations, scores, labels, threshold=0.6):
	index = np.nonzero(np.logical_and(scores > threshold, labels > 0))
	localizations = localizations * 512
	localizations = np.int32(localizations)
	mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
	index=np.array(index)
	index=np.reshape(index,[-1])
	count=0
	for i in range(len(index)):
		idx = index[i]
		loc = localizations[idx,:]
		# if labels[idx]==1:
		count += 1
		red_color = int(scores[idx]*255)
		colors=[0,0,0]
		colors[labels[idx]-1]=red_color
		scalar = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 5, 200), (23, 56, 78)]
		cv2.rectangle(img, (loc[1], loc[0]), (loc[3], loc[2]), scalar[labels[idx] - 1], 2)
		cv2.rectangle(img, (loc[1] - 1, loc[0]), (loc[1] + 60, loc[0] - 18), scalar[labels[idx] - 1], cv2.FILLED)
		cv2.putText(img, str(labels[idx]) + ':%.2f' % scores[idx], (loc[1] + 2, loc[0] - 3), cv2.FONT_HERSHEY_COMPLEX,
					.5, (255, 255, 255))

		mask[loc[0]:loc[2], loc[1]:loc[3]] = 255
	return img, count, mask



with tf.Graph().as_default():
	ssd_nets = SSDNet()
	ssd_anchors = ssd_nets.place_anchors()
	test_img_placeholder = tf.placeholder(tf.float32, [1, None, None, channels])
	test_img = data_utils.preprocess_test(test_img_placeholder)
	test_op = ssd_nets.test_op(test_img, ssd_anchors, reuse=False)

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	if 'checkpoint' in os.listdir(FLAGS.model_path):
		print('restore from last model....')
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('restore ' + ckpt.model_checkpoint_path)


	img_id = 0
	for i, img_file in enumerate(img_files):
		print(img_file)
		img_name = os.path.basename(img_file).split('.tiff')[0]

		if not os.path.exists(os.path.join(OutPath, img_name)):
			os.makedirs(os.path.join(OutPath, img_name))
		if not os.path.exists(os.path.join(OutPath, img_name, 'source_img')):
			os.makedirs(os.path.join(OutPath, img_name, 'source_img'))
		# full_img=gdal_utils.read_full_image(img_file,data_format='NUMPY_FORMAT')
		height, width, _ = gdal_utils.get_image_shape(img_file)
		pbar = tqdm.tqdm(total=height*width//(512*512))
		# result_img = np.zeros((height, width, 3), np.uint8)
		x = 0
		y = 0
		print(height, width)
		result_mask = np.zeros((height, width), np.uint8)
		while x <height:
			part_height = 512
			x = height - part_height if part_height + x > height else x
			result_scores = []
			result_localizations = []
			result_labels = []
			y = 0
			while y < width:
				y_flag = False
				pbar.update(1)
				part_width = 512
				# if y+part_width>width:
				# 	y_flag=True
				y = width - part_width if y + part_width > width else y

				# print(x,y,part_height,part_width)
				# part_img=full_img[x:x+part_height,y:y+part_width,:]

				part_img_source = gdal_utils.read_image(img_file, y, x, part_height, part_width,
														data_format='NUMPY_FORMAT',
														as_rgb=(not channels==4),as_8bit=False)	# if 4-channel: as_rgb=False
				# part_img = gdal_utils.swap_band(part_img_source)	# band-
				part_img = np.expand_dims(part_img_source, 0)
				part_img = part_img.astype(np.float32)/scale_factor
				[img, localization, scores, labels, index, _] = sess.run(test_op, feed_dict={test_img_placeholder: part_img})

				part_img_src = np.squeeze(part_img, axis=0)
				# part_img_src = misc.imresize(part_img_src, [512, 512])/255.
				part_img_src = part_img_src[:, :, 0:channels]
				part_img_src = part_img_src*255.
				part_img_src = part_img_src.astype('uint8')
				cv2.cvtColor(part_img_src, cv2.COLOR_RGB2BGR)

				part_img_result = part_img_src.copy()
				part_img_result, count, mask = draw_boxes(part_img_result, localization[index, :], scores[index],
														  labels[index], threshold=0.5)
				part_img_result = misc.imresize(part_img_result, [part_height, part_width])

				part_mask_result = misc.imresize(mask, [part_height, part_width], interp='nearest')


				if count > 0:
					# part_img_result = Image.fromarray(part_img_result)
					cv2.imencode('.jpg', part_img_result)[1].tofile(os.path.join(OutPath, img_name, '%s_%d_%d.jpg' % (img_name, y, x)))
					cv2.imencode('.jpg', part_img_src)[1].tofile(os.path.join(OutPath, img_name, 'source_img', r'%s_%d_%d.jpg' % (img_name, y, x)))
					# cv2.imwrite(os.path.join(OutPath,
					# 						 r'%s\%s_%d_%d_img.jpg' % (img_name, img_name, y,x)),
					# 						 gdal_utils.swap_band(part_img_source))		# band-

				# result_img[x:x+part_height,y:y+part_width,:]= gdal_utils.swap_band(result_pile_img)
				result_mask[x:x+part_height, y:y+part_width] = part_mask_result
				img_id+=1
				y+=512
			x+=512

		gdal_utils.save_full_image(os.path.join(OutPath, img_name, img_name, '%s_mask.tiff' % (img_name)),
		                           result_mask, data_format='NUMPY_FORMAT',
		                           geoTranfsorm=gdal_utils.get_geoTransform(img_file),  # band-
		                           proj=gdal_utils.get_projection(img_file))
		pbar.close()






