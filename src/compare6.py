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
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import facenet
import align.detect_face

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess
gpu_memory_fraction = 0.3

def main():
    model = "20190128-123456/3000w-train.pb"
    traindata_path = "../data/gump"
    feature_files = []
    face_label = []
    face_detection = Detection()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for images in os.listdir(traindata_path):
                print(images)
                filename = os.path.splitext(os.path.split(images)[1])[0]
                image_path = traindata_path + "/" + images
                images = face_detection.find_faces(image_path)
                if images is not None:
                    face_label.append(filename)
                    # Run forward pass to calculate embeddings
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    feature_files.append(emb)
                else:
                    print('no find face')
            write_file = open('20190128-123456/knn_classifier.pkl', 'wb')
            pickle.dump(feature_files, write_file, -1)
            pickle.dump(face_label, write_file, -1)
            write_file.close()
            print('total_num:',len(os.listdir(traindata_path)))
            print('align_num:',len(face_label))
            print('End')


class Detection:
    minsize = 40  # minimum size of face
    threshold = [0.8, 0.9, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=112, face_crop_margin=0):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)


    def find_faces(self,image_paths):
        img = misc.imread(os.path.expanduser(image_paths), mode='RGB')
        _bbox = None
        _landmark = None
        bounding_boxes, points = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        img_list = []
        max_Aera = 0
        if nrof_faces > 0:
            if nrof_faces == 1:
                bindex = 0
                _bbox = bounding_boxes[bindex, 0:4]
                _landmark = points[:, bindex].reshape((2, 5)).T
                warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                # cv2.imwrite('1.jpg',warped)
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
            else:
                for i in range(nrof_faces):
                    _bbox = bounding_boxes[i, 0:4]
                    if _bbox[2]*_bbox[3] > max_Aera:
                        max_Aera = _bbox[2]*_bbox[3]
                        _landmark = points[:, i].reshape((2, 5)).T
                        warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
        else:
            return None
        images = np.stack(img_list)
        return images


if __name__ == '__main__':
    main()
