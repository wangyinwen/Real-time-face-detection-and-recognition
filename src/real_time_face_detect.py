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
import argparse
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import face


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            cv2.putText(frame, "hu", (face_bb[0], face_bb[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        thickness=2, lineType=2)



def main():
    # print(name)
    frame_interval = 3  # Number of frames after which to run face detection#抽帧检测
    fps_display_interval = 0.1  # seconds
    frame_rate = 0
    frame_count = 0
    # video_capture = cv2.VideoCapture("rtsp://admin:admin@192.168.2.166:554/video1")
    video_capture = cv2.VideoCapture(0)
    width, height = video_capture.get(3), video_capture.get(4)
    print('分辨率：')
    print(width, height)
    face_detection = face.Detection()
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_detection.find_faces(frame)
            # cv2.imwrite("3.jpg",frame)
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate)

        frame_count += 1
        # img_zo = cv2.resize(frame, (int(width) // 3, int(height) // 3), interpolation=cv2.INTER_AREA)
        img_zo = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        # cv2.namedWindow("Video", cv2.CV_WINDOW_NORMAL)  # [2]创建图片的显示窗口
        srcImg = cv2.imread("2.png")
        # cv2.moveWindow("[ROIImg]", 100, 100)
        # cv2.imshow("[ROIImg]", srcImg)
        srcImg = cv2.resize(srcImg, (100, 100), interpolation=cv2.INTER_AREA)
        img_zo[0:100, 0:100] = srcImg
        # cv2.moveWindow("Video", 100, 100)
        cv2.imshow('Video', img_zo)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--debug', action='store_true',
#                         help='Enable some debug outputs.')
#     return parser.parse_args(argv)


if __name__ == '__main__':
    main()
