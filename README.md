# Object (Mobile Phone) Detection Using TensorFlow Object Detection API

Developed a model using TensorFlow Object Detection API to detect object (mobile phone) with location in an image. Preprocessed dataset images with annotations into TF Records and trained on SSD MobileNet using transfer learning.

These are the files you need mainly for preprocessing data, defining hyper parameters such as batch size and number of epochs, training and testing. The test output are bounding boxes that determines the object (mobile, or any other object(s) in your case) by specifying four coordinates and its class label (mobile in my case).  

The coordinates are actually 4 numbers that respectively corresponds to:
1) normalized center x coordinate
2) normalized center y coordinate
3) normalized width
4) normalized height
