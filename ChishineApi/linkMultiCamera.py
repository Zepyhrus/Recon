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
import multiprocessing
    
if __name__ == '__main__':
    KEY_q = 0x71
    KEY_add = 0x31
    KEY_sub = 0x32
    
    openni2.initialize()

    list = openni2.Device.enumerate_uris()
    for uri in list: 
        dev = openni2.Device.open_file(uri)
        print('get_device_info().uri =',dev.get_device_info().uri)
        if dev.has_sensor(openni2.SENSOR_DEPTH) :
            stream = dev.create_stream(openni2.SENSOR_DEPTH)                             
            stream.start()        
            count  = 0
            while count < 20:
                frame = stream.read_frame()
                frame_data = np.array(frame.get_buffer_as_uint16()).reshape([frame.height, frame.width])	            
                cv2.imshow('depth', frame_data)
                cv2.waitKey(34)
                #time.sleep(1)
                count = count + 1
                if count == 20:
                    cv2.destroyAllWindows();  
            stream.close()
        else:
            print("Device does not have depth sensor!")
        time.sleep(1)
        dev.close()
       