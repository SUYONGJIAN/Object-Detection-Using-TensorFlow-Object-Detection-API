# Object (Mobile Phone) Detection Using TensorFlow Object Detection API

Developed a model using TensorFlow Object Detection API to detect object (mobile phone) with location in an image. Preprocessed dataset images with annotations into TF Records and trained on SSD MobileNet using transfer learning.

These are the files you need mainly for preprocessing data, defining hyper parameters such as batch size and number of epochs, training and testing. The test output are bounding boxes that determines the object (mobile, or any other object(s) in your case) by specifying four coordinates and its class label (mobile in my case).  
The coordinates are actually 4 numbers that respectively corresponds to:
1) normalized center x coordinate
2) normalized center y coordinate
3) normalized width
4) normalized height

For the detailed explaination about the data, project and how i implemented it, refer to my video [here](https://www.youtube.com/watch?v=ZxYBdLQgGbw). (Pardon my voice lol)

Please note that you need to download [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). It also has good documentation, installation instructions and tutorials. Also if you want video tutorials for instructions and how to get started, kindly refer to this YouTube playlist by Sentdex [here](https://www.youtube.com/watch?v=COlbP62-B-U&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku). I have largely followed his tutorials and TensorFlow Object Detection API documentations and instructions for this project.

