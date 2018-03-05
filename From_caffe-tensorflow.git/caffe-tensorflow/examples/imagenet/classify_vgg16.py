# -*- coding: utf-8 -*
#!/usr/bin/env python
import argparse
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import os.path as osp
import sys

import models
import dataset
import time
from datetime import datetime

def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open('imagenet-classes.txt', 'rb') as infile:
        class_labels = map(str.strip, infile.readlines())
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    # Display the results
    #print(class_labels)
    #print(image_paths)
    #print(probs)
    #print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    #print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        #print('{:20} {:30} {} %'.format(img_name, class_name, confidence))


def classify(model_data_path, image_pathFolder):
    '''Classify the given images using VGG16.'''

    #print(model_data_path)
    #print(image_pathFolder)
    image_paths=[]
    for filename in os.listdir(image_pathFolder):
        image_paths.append(image_pathFolder+filename) 
    #print(image_paths)

    # Get the data specifications for the VggNet model
    spec = models.get_data_spec(model_class=models.VGG16)
    ##print(spec)
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))
    #print(input_node)

    # Construct the network
    net = models.VGG16({'data': input_node})
    #print("net---------------------")

    # Create an image producer (loads and processes images in parallel)
    image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec)
    #print(image_producer)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #tf.ConfigProto()
    log_device_placement=True # 是否打印设备分配日志
    allow_soft_placement=False # 如果你指定的设备不存在，允许TF自动分配设备
    config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)
    with tf.Session(config=config) as sesh:
        #print('start  -----------------------------------------------------------------%s' % datetime.now())
        #sesh.run(tf.global_variables_initializer())
        # Start the image processing workers
        coordinator = tf.train.Coordinator()
        threads = image_producer.start(session=sesh, coordinator=coordinator)

        # Load the converted parameters
        #print('Loading the model -----------------------------------------------------------------%s' % datetime.now())
        net.load(model_data_path, sesh)


        # Load the input image
        #print('Loading the images-----------------------------------------------------------------%s' % datetime.now())
        indices, input_images = image_producer.get(sesh)

        # Perform a forward pass through the network to get the class probabilities
        print('Classifying        -----------------------------------------------------------------%s' % datetime.now())
        probs = sesh.run(net.get_output(), feed_dict={input_node: input_images})
        print('Classifying END    -----------------------------------------------------------------%s' % datetime.now())
        display_results([image_paths[i] for i in indices], probs)

        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the VGG16 model')
    parser.add_argument('image_paths', help='One or more images to classify')
    # nargs='+',
    args = parser.parse_args()
    #print(parser)
    ##print(args)
    # Classify the image
    classify(args.model_path, args.image_paths)


if __name__ == '__main__':
    main()
