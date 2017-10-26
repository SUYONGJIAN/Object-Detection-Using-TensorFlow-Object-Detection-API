'''
This python file contains code to load the trained model and test it using an image.
After testing, it will show the image with the object (mobile) highlighted with green box and will output the
    normalized x and y coordinates of the center of the object (mobile)

This code has been referred from object_detection_tutorial.ipynb from the folder object_detection
'''
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def main(args):
    if len(args) == 1:
        print("Please enter trained model export folder as argument followed by test image argument, using full path")

    elif len(args) > 3:
        print("Expected two parameters for training folder path, Please enter trained model export folder as argument followed by test image argument, using full path")

    else:
        try:
            trained_model_name = args[1]
            test_image = args[2]
            if trained_model_name[-1] == '/':
                trained_model_name = trained_model_name[0:-1]

            MODEL_NAME = trained_model_name
            PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
            PATH_TO_LABELS = os.path.join('find_phone', 'object-detection.pbtxt')
            NUM_CLASSES = 1

            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()

                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')


            # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)


            def load_image_into_numpy_array(image):
                (im_width, im_height) = image.size
                return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

            # Detection
            #PATH_TO_TEST_IMAGES_DIR = test_image
            TEST_IMAGE_PATHS = [ test_image ]

            # Size, in inches, of the output images.
            IMAGE_SIZE = (12, 8)

            with detection_graph.as_default():
                with tf.Session(graph=detection_graph) as sess:
                    # Definite input and output Tensors for detection_graph
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')       # Each box represents a part of the image where a particular object was detected.
                    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')     # Each score represent how level of confidence for each of the objects.
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')   # Score is shown on the result image, together with the class label.
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    for image_path in TEST_IMAGE_PATHS:
                        image = Image.open(image_path)
                        image_np = load_image_into_numpy_array(image)

                        image_np_expanded = np.expand_dims(image_np, axis=0)  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

                        # Actual detection.
                        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
                        b = np.squeeze(boxes)
                        b = b[0]
                        print("{:.4f}".format((b[1]+b[3])/2) + " " + "{:.4f}".format((b[0]+b[2])/2)) #this statement will print the normalized x and y coordinate of the center of the mobile

                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=4)

                        plt.figure(figsize=IMAGE_SIZE)
                        plt.imshow(image_np)
                        plt.show()
        except:
            print("Path is invalid, or Please enter trained model export folder as argument followed by test image argument, using full path")

if __name__ == '__main__':
    main(sys.argv)
