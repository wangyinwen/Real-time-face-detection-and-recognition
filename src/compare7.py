"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from numpy import *
import tensorflow as tf
import numpy as np
import sys
import os
import facenet
import lfw

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import cv2
gpu_memory_fraction = 0.5

def main():
    model = "20190128-123456/3001w-train.pb"
    pairs = lfw.read_pairs('../database/npairs.txt')
    paths, actual_issame = lfw.get_paths('F:/aface-clean-rename2', pairs)
    true_pairs = []
    false_pairs = []
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Load the model
            facenet.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for i in range(len(pairs)):
                print(paths[2 * i])
                img1 = cv2.imread(paths[2 * i])
                img2 = cv2.imread(paths[2 * i + 1])
                frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                f1 = []
                f2 = []
                prewhitened1 = facenet.prewhiten(frame1)
                f1.append(prewhitened1)
                prewhitened2 = facenet.prewhiten(frame2)
                f2.append(prewhitened2)
                feed_dict = {images_placeholder: f1, phase_train_placeholder: False}
                emb1 = sess.run(embeddings, feed_dict=feed_dict)
                feed_dict = {images_placeholder: f2, phase_train_placeholder: False}
                emb2 = sess.run(embeddings, feed_dict=feed_dict)
                sim = dot(emb1, emb2.T)
                sim = sim * 0.5 + 0.5
                if actual_issame[i] == True:
                    true_pairs.append(sim)
                else:
                    false_pairs.append(sim)
    thresholds = np.arange(0, 1, 0.01)
    total_ratelist = []
    total_rightnum = 0
    total_rightnum2 = 0
    f = open('../database/restlt.txt', 'w')
    f.truncate()
    for threshold in thresholds:
        # print(threshold)
        true_num = 0
        false_num = 0
        for i in range(len(true_pairs)):
            if float(true_pairs[i]) > threshold:
                true_num = true_num + 1
        for j in range(len(false_pairs)):
            if float(false_pairs[j]) < threshold:
                false_num = false_num + 1
        if (true_num + false_num) >= total_rightnum2:
            total_rightnum2 = true_num + false_num
            print(total_rightnum2)
            print(threshold, ':', format(float(total_rightnum2)/float(len(pairs)), '.4f'))
        total_rightnum = true_num + false_num
        f.writelines([str(threshold), ' ',str(format(float(total_rightnum) / float(len(pairs)), '.4f')), '\n'])
    f.close()

if __name__ == '__main__':
    main()
