# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
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
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
import face
import sys
import facegui
import facenet
import align.detect_face
import numpy
import pickle
from scipy import misc
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
add_name = ''
import face_preprocess
from PyQt5.QtWidgets import QMessageBox


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        # self.face_recognition = face.Recognition()
        self.face_detection = Detection()
        self.face_detection_capture = face.Detection()
        self.timer_camera = QtCore.QTimer()
        self.timer_camera_capture = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.opencamera = QtWidgets.QPushButton(u'人脸识别')
        self.addface = QtWidgets.QPushButton(u'建库')
        self.captureface = QtWidgets.QPushButton(u'采集人脸')
        self.saveface = QtWidgets.QPushButton(u'保存人脸')
        self.opencamera.setMinimumHeight(50)
        self.addface.setMinimumHeight(50)
        self.captureface.setMinimumHeight(50)
        self.saveface.setMinimumHeight(50)
        self.lineEdit = QtWidgets.QLineEdit(self)  # 创建 QLineEdit
        self.lineEdit.textChanged.connect(self.text_changed)
        self.lineEdit.setMinimumHeight(50)

        # self.opencamera.move(10, 30)
        # self.captureface.move(10, 50)
        self.lineEdit.move(15,350)

        # 信息显示
        self.showcamera = QtWidgets.QLabel()
        # self.label_move = QtWidgets.QLabel()
        self.lineEdit.setFixedSize(70, 30)

        self.showcamera.setFixedSize(641, 481)
        self.showcamera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.opencamera)
        self.__layout_fun_button.addWidget(self.addface)
        self.__layout_fun_button.addWidget(self.captureface)
        self.__layout_fun_button.addWidget(self.saveface)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.showcamera)

        self.setLayout(self.__layout_main)
        # self.label_move.raise_()
        self.setWindowTitle(u'FaceRec')

    def slot_init(self):
        self.opencamera.clicked.connect(self.button_open_camera_click)
        self.addface.clicked.connect(self.button_add_face_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera_capture.timeout.connect(self.capture_camera)
        self.captureface.clicked.connect(self.button_capture_face_click)
        self.saveface.clicked.connect(self.save_face_click)

    def text_changed(self):  # 这个函数能够实时打印文本框的内容
        global add_name
        add_name = self.lineEdit.text()
        print(u'文本框此刻输入的内容是：%s' % add_name)

    def button_open_camera_click(self):
        self.timer_camera_capture.stop()
        self.cap.release()
        self.showcamera.clear()
        self.face_recognition = face.Recognition()
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)

                self.opencamera.setText(u'关闭识别')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.showcamera.clear()
            self.opencamera.setText(u'人脸识别')

    def show_camera(self):
        flag, self.image = self.cap.read()
        # face = self.face_detect.align(self.image)
        # if face:
        #     pass
        show = cv2.resize(self.image, (640, 480))
        # face_detection = self.face.Detection()
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        faces = self.face_recognition.identify(show)
        if faces is not None:
            if faces is not None:
                img_PIL = Image.fromarray(show)
                font = ImageFont.truetype('simsun.ttc', 40)
                # 字体颜色
                fillColor1 = (255, 0, 0)
                fillColor2 = (0, 255, 0)
                draw = ImageDraw.Draw(img_PIL)
                for face in faces:
                    face_bb = face.bounding_box.astype(int)
                    draw.line([face_bb[0], face_bb[1], face_bb[2], face_bb[1]], "green")
                    draw.line([face_bb[0], face_bb[1], face_bb[0], face_bb[3]], fill=128)
                    draw.line([face_bb[0], face_bb[3], face_bb[2], face_bb[3]], "yellow")
                    draw.line([face_bb[2], face_bb[1], face_bb[2], face_bb[3]], "black")
                    if face.name is not None:
                        if face.name == 'unknown':
                            draw.text((face_bb[0], face_bb[1]), '陌生人', font=font, fill=fillColor2)
                        else:
                            draw.text((face_bb[0], face_bb[1]), face.name, font=font, fill=fillColor1)
            show = numpy.asarray(img_PIL)
        # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.showcamera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def button_add_face_click(self):
        self.timer_camera_capture.stop()
        self.cap.release()
        self.showcamera.clear()
        model = "20190128-123456/3001w-train.pb"
        traindata_path = "../data/gump"
        feature_files = []
        face_label = []
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
                    images = self.face_detection.find_faces(image_path)
                    if images is not None:
                        face_label.append(filename)
                        # Run forward pass to calculate embeddings
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        print(emb)
                        feature_files.append(emb)
                    else:
                        print('no find face')
                write_file = open('20190128-123456/knn_classifier.pkl', 'wb')
                pickle.dump(feature_files, write_file, -1)
                pickle.dump(face_label, write_file, -1)
                write_file.close()
        reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "建库",
                                        "建库完成",
                                        QMessageBox.Yes | QMessageBox.No)

    def button_capture_face_click(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.timer_camera_capture.start(30)

    def capture_camera(self):
        flag, self.images = self.cap.read()
        self.images = cv2.cvtColor(self.images, cv2.COLOR_BGR2RGB)
        show_images = self.images
        faces = self.face_detection_capture.find_faces(show_images)
        if faces is not None:
            for face in faces:
                face_bb = face.bounding_box.astype(int)
                cv2.rectangle(show_images,
                              (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                              (0, 255, 0), 2)
        show_images = numpy.asarray(show_images)
        showImage = QtGui.QImage(show_images.data, show_images.shape[1], show_images.shape[0], QtGui.QImage.Format_RGB888)
        self.showcamera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def save_face_click(self):
        global add_name
        imagepath = os.sep.join(['../data/gump/', add_name + '.jpg'])
        print('faceID is:',add_name)
        if add_name == '':
            reply = QMessageBox.information(self,  # 使用infomation信息框
                                            "人脸ID",
                                            "请在文本框输入人脸的ID",
                                            QMessageBox.Yes | QMessageBox.No)
        else:
            self.images = cv2.cvtColor(self.images, cv2.COLOR_RGB2BGR)
            cv2.imencode(add_name + '.jpg', self.images)[1].tofile(imagepath)
            # cv2.imwrite('../data/gump/' + '胡亚洲' + '.jpg', self.images)

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


def add_overlays(frame, faces):
    if faces is not None:
        img_PIL = Image.fromarray(frame)
        font = ImageFont.truetype('simsun.ttc', 40)
        # 字体颜色
        fillColor1 = (255, 0, 0)
        fillColor2 = (0, 255, 0)
        draw = ImageDraw.Draw(img_PIL)
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            draw.line([face_bb[0], face_bb[1], face_bb[2], face_bb[1]], "green")
            draw.line([face_bb[0], face_bb[1], face_bb[0], face_bb[3]], fill=128)
            draw.line([face_bb[0], face_bb[3], face_bb[2], face_bb[3]], "yellow")
            draw.line([face_bb[2], face_bb[1], face_bb[2], face_bb[3]], "black")
            if face.name is not None:
                if face.name == 'unknown':
                    draw.text((face_bb[0], face_bb[1]), '陌生人', font=font, fill=fillColor2)
                else:
                    draw.text((face_bb[0], face_bb[1]), face.name, font=font, fill=fillColor1)
        frame = numpy.asarray(img_PIL)
        return frame


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
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image_paths):
        img = misc.imread(os.path.expanduser(image_paths), mode='RGB')
        _bbox = None
        _landmark = None
        bounding_boxes, points = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                               self.threshold, self.factor)
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
                    if _bbox[2] * _bbox[3] > max_Aera:
                        max_Aera = _bbox[2] * _bbox[3]
                        _landmark = points[:, i].reshape((2, 5)).T
                        warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
        else:
            return None
        images = np.stack(img_list)
        return images


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
