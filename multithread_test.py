import tensorflow as tf
import os
import glob
import gdal_utils
import path_utils
import tqdm
import time
from threading import Thread
import pickle

from config import FLAGS
from nets.ssd_vgg_512_Multi import SSDNet_vgg_512 as  SSDNet
import dataset.dataSet_utils as data_utils
from input import DataInputTest
from input import cal_result

slim=tf.contrib.slim

OutPath=r'F:\测试'

if not os.path.exists(OutPath):
	os.mkdir(OutPath)

with tf.Graph().as_default():
	ssd_nets = SSDNet()
	ssd_anchors = ssd_nets.place_anchors()
	batch_size = FLAGS.test_batch_size
	test_img_placeholder=tf.placeholder(tf.float32,[None,None,None,3])
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

	InPath=r'F:\测试'
	img_files=glob.glob(os.path.join(InPath, r'*.tiff'))
	img_id=0
	for i, img_file in enumerate(img_files):
		print(img_file)
		file_name = img_file.split("\\")[-1]
		start = time.time()
		loc_rst = []
		score_rst = []
		label_rst = []
		height, width, _ = gdal_utils.get_image_shape(img_file)
		img_info = [height, width]

		for _, data in DataInputTest(batch_size, img_file):
			batch = data[0]
			offsets = data[1]
			[_, localizations, scores, labels, index, _] = sess.run(test_op,
																	 feed_dict={test_img_placeholder: batch})
			loc_info, score_info, label_info = cal_result(localizations, scores, labels, offsets, threshold=0.5)
			loc_rst+=loc_info
			score_rst+=score_info
			label_rst+=label_info

		assert len(loc_rst) == len(score_rst)
		assert len(loc_rst) == len(label_rst)

		out_file = os.path.join(OutPath, file_name[:-4]+".pkl")
		with open(out_file, 'wb') as f:
			pickle.dump(file_name, f, pickle.HIGHEST_PROTOCOL)
			pickle.dump(img_info, f, pickle.HIGHEST_PROTOCOL)
			pickle.dump(loc_rst, f, pickle.HIGHEST_PROTOCOL)
			pickle.dump(score_rst, f, pickle.HIGHEST_PROTOCOL)
			pickle.dump(label_rst, f, pickle.HIGHEST_PROTOCOL)

		end = time.time()
		print("detection time:%d"%(end-start))







