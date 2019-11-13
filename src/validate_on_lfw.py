"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
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

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate


def main(args):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
            print(len(pairs))  # 一共有6000对
            # 读入后如[['Abel_Pacheco','1','4']]

            # Get the paths for the corresponding images
            # 获取文件路径和是否匹配关系对
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
            # print(len(actual_issame))#len(actual_issame) = 6000
            # args.lfw_file_ext表示图像的格式，默认png格式
            # actual_issame为标志位，如果是一对actual_issame=true 否则等于false
            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]  # 特征向量维数128
            print(embedding_size)
            # Run forward pass to calculate embeddings
            # 3. 使用前向传播验证
            print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size  # default=100
            nrof_images = len(paths)  # 测试的人脸比对次数6000，len（paths）=12000
            # print(paths)表示paths为读取图片的路径
            # print(nrof_images)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))  ## 总共批次数，ceil() 函数返回数字的上入整数。结果为120
            emb_array = np.zeros((nrof_images, embedding_size))  # 声明固定大小的空矩阵12000*128
            for i in range(nrof_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)  # 加载图片做一些变换crop和flip
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)  # 特征向量push
                print(('WanCheng: %d' % (i + 1)))
            # 十折交叉验证(10-fold cross validation)，精度测试方法。数据集分成10份，
            # 轮流将其中9份做训练集，1份做测试保，10次结果均值作算法精度估计
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                                                                 actual_issame,
                                                                 nrof_folds=args.lfw_nrof_folds)  # nrof_folds =10

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('lfw_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
