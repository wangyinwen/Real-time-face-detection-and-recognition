"""Helper for evaluation on the Labeled Faces in the Wild dataset
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

import os
import numpy as np
import facenet


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)  # 0.01表示步长
    # 四个点原来是取奇数和偶数列
    embeddings1 = embeddings[0::2]  # 取出奇数行的特征，取出偶数行的特征
    embeddings2 = embeddings[1::2]
    # print(len(embeddings1))
    # print('wokao')
    # print(len(embeddings2))
    tpr, fpr, accuracy, threshold_acc = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                                              np.asarray(actual_issame),
                                                              nrof_folds=nrof_folds)  # np.asarray将列表转换为数组
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, threshold_acc, embeddings1, embeddings2,
                                              np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


# 在lfw数据集上测试
# def get_paths(lfw_dir, pairs):
#     nrof_skipped_pairs = 0
#     path_list = []
#     issame_list = []
#     file_ext = 'jpg'
#     for pair in pairs:
#         if len(pair) == 3:
#             path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
#             path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
#             issame = True    #读取每一行，如果行的数据为3个则表示是同一个人，如果是四个则表示不是同一个人
#         elif len(pair) == 4:
#             path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
#             path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
#             issame = False
#         if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
#             path_list += (path0,path1)
#             issame_list.append(issame)
#         else:
#             nrof_skipped_pairs += 1
#     if nrof_skipped_pairs>0:
#         print('Skipped %d image pairs' % nrof_skipped_pairs)
#
#     return path_list, issame_list

# 在自制的数据集上测试
def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        print(len(pair))
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], '%s' % pair[1])
            # print(path0)
            path1 = os.path.join(lfw_dir, pair[0], '%s' % pair[2])
            issame = True  # 读取每一行，如果行的数据为3个则表示是同一个人，如果是四个则表示不是同一个人
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], '%s' % pair[1])
            path1 = os.path.join(lfw_dir, pair[2], '%s' % pair[3])
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            print("miss:")
            print(path0)
            print(path1)
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)
