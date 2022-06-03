# -*- coding: utf-8 -*-
# @File   : depthimage_display.py
# @Date   : 2020/07/14
# @Author : PengLei
# @Mail   : penglei@chishine3d.com
# @bref   : This is a depth stream sample.
#           press Key 'q' to exit the sample
#           press Key '1' to increase gain
#           press Key '2' to decrease gain
import ctypes
from openni import openni2
from openni import _openni2
import numpy as np
import cv2
    
if __name__ == '__main__':
    KEY_q = 0x71
    KEY_add = 0x31
    KEY_sub = 0x32
    
    openni2.initialize()
    dev = openni2.Device.open_any()

    if dev.has_sensor(openni2.SENSOR_DEPTH) :
        print("This is a depth stream sample.")
        print("press Key 'q' to exit the sample")
        print("press Key '1' to increase gain")
        print("press Key '2' to decrease gain")

        stream = dev.create_stream(openni2.SENSOR_DEPTH)
        sensor_info = stream.get_sensor_info()
        print(sensor_info.videoModes[-1])
        stream.set_video_mode(sensor_info.videoModes[len(sensor_info.videoModes)-1])
        stream.start()

        stream.set_property(_openni2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, False)
        
        while True:
            frame = stream.read_frame()
            frame_data = np.array(frame.get_buffer_as_uint16()).reshape([frame.height, frame.width])
            cv2.imshow('depth', frame_data)
            
            key = int(cv2.waitKey(10))
            if key == KEY_q:
                break
            if key == KEY_sub:
                gain = stream.get_property(_openni2.ONI_STREAM_PROPERTY_GAIN, ctypes.c_uint32)
                gain = gain.value - 1
                stream.set_property(_openni2.ONI_STREAM_PROPERTY_GAIN, gain)
            if key == KEY_add:
                gain = stream.get_property(_openni2.ONI_STREAM_PROPERTY_GAIN, ctypes.c_uint32)
                gain = gain.value + 1
                stream.set_property(_openni2.ONI_STREAM_PROPERTY_GAIN, gain)
                
        stream.stop()
    else:
        print("Device does not have depth sensor!")
    dev.close()
