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
import time
import cv2
import face
import numpy
from PIL import Image, ImageDraw, ImageFont

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
            draw.line([face_bb[0],  face_bb[1], face_bb[2], face_bb[1]], "green")
            draw.line([face_bb[0],  face_bb[1], face_bb[0], face_bb[3]], fill=128)
            draw.line([face_bb[0], face_bb[3], face_bb[2], face_bb[3]], "yellow")
            draw.line([face_bb[2], face_bb[1], face_bb[2], face_bb[3]], "black")
            if face.name is not None:
                if face.name == 'unknown':
                    draw.text((face_bb[0], face_bb[1]), '陌生人', font=font, fill=fillColor2)
                else:
                    draw.text((face_bb[0], face_bb[1]), face.name, font=font, fill=fillColor1)
        frame = numpy.asarray(img_PIL)
        return frame


def main():
    # video_capture = cv2.VideoCapture("rtsp://admin:12345678hu@192.168.0.100/Streaming/Channels/1")
    # video_capture = cv2.VideoCapture("rtsp://admin:12345678hu@192.168.0.100:80/h264/ch1/main/av_stream")
    face_recognition = face.Recognition()
    frame_interval = 5  # Number of frames after which to run face detection#抽帧检测
    frame_count = 0
    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)
            frame_count = 0
        frame_count += 1
        # Check our current fps
        frame = add_overlays(frame, faces)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
