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
import csdevice as cs
import numpy as np
import cv2
import time
    
if __name__ == '__main__':
    KEY_q = 0x71
    KEY_add = 0x31
    KEY_sub = 0x32
    
    openni2.initialize()
    dev = openni2.Device.open_any()

    if dev.has_sensor(openni2.SENSOR_DEPTH) :

        stream = dev.create_stream(openni2.SENSOR_DEPTH)
        #sensor_info = stream.get_sensor_info()
        #stream.set_video_mode(sensor_info.videoModes[len(sensor_info.videoModes)-1])
        stream.start()

        while True:
	    #get trigger mode 0-off 1-external trigger mode 2-software trigger mode
            trMode = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_TRIGGER_MODE, ctypes.c_int)
            print ('trigger mode:',trMode.value)
	    #set trigger model is software trigger
            trMode.value = 2
            stream.set_property(cs.CS_PROPERTY_STREAM_EXT_TRIGGER_MODE, trMode.value);
            frame = stream.read_frame()

            frame_data = np.array(frame.get_buffer_as_uint16()).reshape([frame.height, frame.width])
            #cv2.imshow('depth', frame_data)
            #key = int(cv2.waitKey(10))
            while 1:           
                time.sleep(5)
		#do softTrigger()
                stream.set_property(cs.CS_PROPERTY_STREAM_EXT_SOFT_TRIGGER,ctypes.c_uint32(0))
        stream.stop()
    else:
        print("Device does not have depth sensor!")
    dev.close()
