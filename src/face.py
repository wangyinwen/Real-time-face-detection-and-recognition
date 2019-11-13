# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
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
from numpy import *
import sys
import numpy as np
import tensorflow as tf
import os
import pickle
import align.detect_face
import facenet
import heapq
import time
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess

gpu_memory_fraction = 0.5
facenet_model_checkpoint = os.path.dirname(__file__) + "/20190128-123456/3001w-train.pb"
classifier_model = os.path.dirname(__file__) + "/20190128-123456/knn_classifier.pkl"  # 模型不同，提取的特征向量不同

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        faces = self.detect.find_faces(image)
        for i, face in enumerate(faces):
            start_time = time.time()
            face.embedding = self.encoder.generate_embedding(face)
            end_time = time.time()
            print(float('%.3f' % (end_time - start_time)))
            face.name = self.identifier.identify(face)

        return faces


class Identifier:
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model = pickle.load(infile)
            self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            label_num = len(self.class_names)
            print('the class is: %s' % label_num)
            cosdist = []
            for i in range(label_num):
                num = np.dot(self.model[i], face.embedding.T)
                sim = 0.5 + 0.5 * num  # 归一化，，余弦距离
                # num = dot(self.model[i], face.embedding)
                # denom = linalg.norm(self.model[i]) * linalg.norm(face.embedding)
                # cos = num / denom  # 余弦值
                # sim = 0.5 + 0.5 * cos  # 归一化，，余弦距离
                cosdist.append(sim)
            print('the max and second cosdis is:')
            print(heapq.nlargest(2, cosdist))  # 返回相似度最大的两个人的相似度
            print('the second similarity is: %s' % self.class_names[cosdist.index(heapq.nlargest(2, cosdist)[1])])
            if max(cosdist) > 0.785:  # webface数据集的模型
                return self.class_names[cosdist.index(max(cosdist))]
            else:
                return "unknown"

class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 80  # minimum size of face
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
    # 对齐的人脸
    def find_faces(self, image):
        faces = []
        bindex = 0
        bounding_boxes, points= align.detect_face.detect_face(image, self.minsize,
                                                self.pnet, self.rnet, self.onet,
                                                self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])

            _bbox = bounding_boxes[bindex, 0:4]
            _landmark = points[:, bindex].reshape((2, 5)).T
            bindex = bindex + 1
            warped = face_preprocess.preprocess(image, bbox=bounding_boxes, landmark=_landmark, image_size='112,112')
            # cv2.imwrite('1.jpg', warped)
            face.image = warped
            faces.append(face)
        return faces
    #返回最大的人脸
    def find_max_faces(self, image):
        faces = []
        num = 0
        bounding_boxes, points= align.detect_face.detect_face(image, self.minsize,
                                                self.pnet, self.rnet, self.onet,
                                                self.threshold, self.factor)
        Area_max = 0
        max_face = []
        num_chips = 0
        for bb in bounding_boxes:
            bounding_box = np.zeros(4, dtype=np.int32)
            bounding_box[0] = bb[0]
            bounding_box[1] = bb[1]
            bounding_box[2] = bb[2]
            bounding_box[3] = bb[3]
            if (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1]) >= Area_max:
                max_face = []
                Area_max = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])
                max_face.append(bounding_box)
                num_chips = num
            num = num + 1
        for bb in max_face:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])

            _bbox = bounding_boxes[num_chips, 0:4]
            _landmark = points[:, num_chips].reshape((2, 5)).T
            warped = face_preprocess.preprocess(image, bbox=_bbox, landmark=_landmark, image_size='112,112')
            face.image = warped
            faces.append(face)
        return faces
