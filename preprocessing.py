'''
This python file contains preprocessing utility functions
'''
import tensorflow as tf
import cv2
import numpy as np
import os
import csv
import io
from PIL import Image
from object_detection.utils import dataset_util

'''
This function loads the training labels from the csv file
	parameters:
		the path of the csv file
	returns:
		the training labels in numpy array and sorted according to filename for consistency
'''
def load_training_labels(labels_path):
	training_labels = []

	with open(labels_path) as f:
		csv_reader = csv.reader(f, delimiter=" ")

		for line in csv_reader:
			temp_line = []
			for i in range(0, len(line)):
				if i == 0:
					splitted = line[i].split(".")
					temp_line.append(int(splitted[0]))
				else:
					temp_line.append(float(line[i]))

			training_labels.append(temp_line)

	training_labels = np.array(training_labels)
	training_labels = training_labels[training_labels[:, 0].argsort()]	# sort according to file name(number)
	return training_labels


'''
This function creates a tf.Example proto from sample image.
referred from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
	parameters:
		delta: the offset from the center in all 4 directions
		cropped_dir: path of the folder containing candidate images for conversion
		image_num: the image number corresponding to the file name
		center_x: x coordinate of the center of mobile
		center_y: y coordinate of the center of mobile
	Returns:
		The created tf.Example.
'''
def create_tf_example(delta, cropped_dir, image_num, center_x, center_y):
	with tf.gfile.GFile(cropped_dir + os.sep + str(image_num) + '.jpg', 'rb') as fid:
		encoded_image_data = fid.read()

	encoded_image_data_io = io.BytesIO(encoded_image_data)
	image = Image.open(encoded_image_data_io)
	width, height = image.size

	filename = str(image_num) + '.jpg'
	image_format = b'jpg'

	xmins = [float(float(center_x - delta) / float(width))]
	xmaxs = [float(float(center_x + delta) / float(width))]
	ymins = [float(float(center_y - delta) / float(height))]
	ymaxs = [float(float(center_y + delta) / float(height))]
	classes_text = ['mobile'.encode('utf8')]
	classes = [1]

	tf_example = tf.train.Example(features=tf.train.Features(feature={
	  'image/height': dataset_util.int64_feature(height),
	  'image/width': dataset_util.int64_feature(width),
	  'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
	  'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
	  'image/encoded': dataset_util.bytes_feature(encoded_image_data),
	  'image/format': dataset_util.bytes_feature(image_format),
	  'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
	  'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
	  'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
	  'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
	  'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
	  'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	return tf_example


'''
This function all the images into tfrecords, which can be used for input in the tensorflow object detection api models.
referred from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
	parameters:
		delta: the offset from the center in all 4 directions
		image_folder_path: path of the folder containing candidate images for conversion
		new_tfrecord_dir_train: name of the training tfrecord to be created
		new_tfrecord_dir_eval: name of the evaluation tfrecord to be created
		training_labels: numpy array containing the training labels
		split_ratio: ratio of samples to be collected into training tfrecord, while the rest will go into evaluation tfrecord
'''
def convert_to_tfrecords(delta, image_folder_path, new_tfrecord_dir_train, new_tfrecord_dir_eval, training_labels, split_ratio=0.8):
	num_files = 0
	for obj in os.listdir(image_folder_path):
		if '.jpg' in str(obj):
			num_files += 1

	num_files_train = int(num_files * split_ratio)
	print('number of training samples:', num_files_train)
	print('number of testing samples:', num_files-num_files_train)

	writer = tf.python_io.TFRecordWriter(image_folder_path + os.sep + new_tfrecord_dir_train)

	for i in range(0, num_files_train):
		training_label = training_labels[i]
		image_num = int(training_label[0])
		center_x = int(490.0 * float(training_label[1]))
		center_y = int(326.0 * float(training_label[2]))
		tf_example = create_tf_example(delta, image_folder_path, str(image_num), center_x, center_y)
		writer.write(tf_example.SerializeToString())

	writer = tf.python_io.TFRecordWriter(image_folder_path + os.sep + new_tfrecord_dir_eval)

	for i in range(num_files_train, num_files):
		training_label = training_labels[i]
		image_num = int(training_label[0])
		center_x = int(490.0 * float(training_label[1]))
		center_y = int(326.0 * float(training_label[2]))
		tf_example = create_tf_example(delta, image_folder_path, str(image_num), center_x, center_y)
		writer.write(tf_example.SerializeToString())


	writer.close()
