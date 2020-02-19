import glob
import os
import gdal_utils
import path_utils
import tqdm
import numpy as np
from scipy import misc
from config import FLAGS
import tensorflow as tf
from nets.ssd_vgg_512_tank import SSDNet_vgg_512 as SSDNet
import dataset.dataSet_utils as data_utils

def draw_boxes(img,localizations,scores,labels,img_name,threshold=0.5):
	index = np.nonzero(np.logical_and(scores > threshold, labels > 0))

	index=np.array(index)
	index=np.reshape(index,[-1])
	count=0
	for i in range(len(index)):
		idx=index[i]
		if labels[idx]==1:
			count+=1
	return count

with tf.Graph().as_default():
	ssd_nets = SSDNet()
	ssd_anchors = ssd_nets.place_anchors()
	test_img_placeholder=tf.placeholder(tf.float32,[1,None,None,3])
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

test_path=r'D:\20190315ziyuan\20190315待检测数据'
img_files=glob.glob(os.path.join(test_path,'*.tiff'))

test_result_path='piles'
if not os.path.exists(test_result_path):
	os.mkdir(test_result_path)
negtive_count=0
for i,img_file in enumerate(img_files):
	img_id = 0
	print(img_file)
	img_name=path_utils.get_filename(img_file,is_suffix=False)
	# if not os.path.exists(os.path.join(test_result_path,img_name)):
	# 	os.mkdir(os.path.join(test_result_path,img_name))
	# full_img=gdal_utils.read_full_image(img_file,data_format='NUMPY_FORMAT')
	height,width,_=gdal_utils.get_image_shape(img_file)
	pbar=tqdm.tqdm(total=height*width//(512*512))
	x=0
	y=0
	print(height,width)
	while x<height:
		part_height=512
		# part_height=height-x if x+part_height>=height else part_height
		x=height-part_height if part_height+x>height else x
		result_scores=[]
		result_localizations=[]
		result_labels=[]
		y=0
		while y<width:
			pbar.update(1)
			part_width=512
			# part_width=width-y if y+part_width>=width else part_width
			y=width-part_width if y+part_width>width else y

			part_img_source=gdal_utils.read_image(img_file,y,x,part_height,part_width,data_format='NUMPY_FORMAT')
			part_img_source=gdal_utils.swap_band(part_img_source)

			part_img = part_img_source.astype(np.float32) / 255.
			part_img=np.expand_dims(part_img,axis=0)
			[img, localization, scores, labels, index, _] = sess.run(test_op,
			                                                         feed_dict={test_img_placeholder: part_img})

			count = draw_boxes(part_img, localization[index, :], scores[index], labels[index],
			                                    os.path.join(test_result_path, ''), threshold=0.1)
			if count>0 or np.mod(negtive_count,50)==1:
				misc.imsave(os.path.join(test_result_path, r'%s_%d.jpg' % (img_name,img_id)),
				                           part_img_source)


			img_id+=1
			y+=512
		x+=512

