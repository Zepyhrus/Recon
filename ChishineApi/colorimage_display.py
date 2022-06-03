# -*- coding: utf-8 -*-
# @File   : colorimage_display.py
# @Date   : 2020/07/14
# @Author : PengLei
# @Mail   : penglei@chishine3d.com
# @bref   : This is a color stream sample.
#           press Key 'q' to exit the sample
#           press Key 'a' to enable auto exposure
#           press Key 'd' to disable auto exposure
import ctypes
from openni import openni2
from openni import _openni2
import numpy as np
import cv2

if __name__ == '__main__':
    KEY_q = 0x71
    KEY_a = 0x61
    KEY_d = 0x64
    
    openni2.initialize()
    dev = openni2.Device.open_any()

    if dev.has_sensor(openni2.SENSOR_COLOR) :
        print("This is a color stream sample.")
        print("press Key 'q' to exit the sample")
        print("press Key '1' to increase gain")
        print("press Key '2' to decrease gain")

        stream = dev.create_stream(openni2.SENSOR_COLOR)
        sensor_info = stream.get_sensor_info()
        
        for videoMode in sensor_info.videoModes:
            if videoMode.pixelFormat == _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888:
                stream.set_video_mode(videoMode)
                break
        stream.start()

        while True:
            frame = stream.read_frame()

            frame_data = np.array(frame.get_buffer_as_triplet()).reshape([frame.height, frame.width, 3])
            R = frame_data[:, :, 0]
            G = frame_data[:, :, 1]
            B = frame_data[:, :, 2]
            frame_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
            
            cv2.imshow('color', frame_data)

            key = int(cv2.waitKey(10))
            if key == KEY_q:
                break
            if key == KEY_a:
                print("enable auto exposure")
                stream.set_property(_openni2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.c_uint32(1))
            if key == KEY_d:
                print("disable auto exposure")
                stream.set_property(_openni2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.c_uint32(0))
        
        stream.stop()
    else:
        print("Device does not have color sensor!")
    dev.close()

