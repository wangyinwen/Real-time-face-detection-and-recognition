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
import cv2
import face
import os
import time

def add_overlays(image, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(image,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                if face.name == 'unknown':
                    cv2.putText(image, face.name, (face_bb[0], face_bb[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                thickness=2, lineType=2)
                else:
                    cv2.putText(image, face.name, (face_bb[0], face_bb[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                thickness=2, lineType=2)


def main():
    testdata_path = '../images'
    face_recognition = face.Recognition()
    start_time = time.time()
    for images in os.listdir(testdata_path):
        print(images)
        filename = os.path.splitext(os.path.split(images)[1])[0]
        file_path = testdata_path + "/" + images
        image = cv2.imread(file_path)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_recognition.identify(frame)
        add_overlays(image, faces)
        cv2.imwrite('../images_result/' + filename + '.jpg', image)
    end_time = time.time()
    spend_time = float('%.2f' % (end_time - start_time))
    print('spend_time:',spend_time)

if __name__ == '__main__':
    main()
