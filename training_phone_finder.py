import numpy as np
import os
import shutil
import sys

from preprocessing import load_training_labels, convert_to_tfrecords

def main(args):
    if len(args) == 1:
        print("Please enter training folder as argument")

    elif len(args) > 2:
        print("Expected one parameter for training folder path")

    else:
        try:
            training_folder = args[1]
            if training_folder[-1] == '/':
                training_folder = training_folder[0:-1]

            image_folder_path = training_folder
            new_tfrecord_dir_train = 'tf_records.record'
            new_tfrecord_dir_eval = 'tf_records_eval.record'
            exported_model_directory = 'exported_model_directory'
            delta = 30	# 30 covers the mobile boundaries in almost all images

            training_labels = load_training_labels(image_folder_path + os.sep + 'labels.txt')
            # convert_to_tfrecords(delta, image_folder_path, new_tfrecord_dir_train, new_tfrecord_dir_eval, training_labels)

            # training:
            # os.system("python3 object_detection/train.py --logtostderr --train_dir="+image_folder_path+" --pipeline_config_path="+image_folder_path+"/ssd_mobilenet_v1_pets.config")

            # removing existing direcrtory of exported model, if exists.
            if os.path.exists(image_folder_path + os.sep + exported_model_directory):
                shutil.rmtree(image_folder_path + os.sep + exported_model_directory)

            # calculate the most recent model.ckpt files, since they will have least loss
            latest_model_num = 0
            candidate_models = []
            for obj in os.listdir(image_folder_path):
                if '.index' in str(obj):
                    temp = obj.split('.')
                    temp = temp[1]
                    temp = temp[5:]
                    candidate_models.append(int(temp))
                    candidate_models.sort()
                    latest_model_num = candidate_models[-1]

            # exporting the trained model (inference graph):
            os.system("python3 object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path "+image_folder_path+"/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix "+image_folder_path+"/model.ckpt-"+str(latest_model_num)+" --output_directory "+image_folder_path+"/exported_model_directory")
            print("Model exported to "+image_folder_path+"/exported_model_directory")

        except:
            print("Please enter full path, or the path is invalid")

if __name__ == '__main__':
    main(sys.argv)
