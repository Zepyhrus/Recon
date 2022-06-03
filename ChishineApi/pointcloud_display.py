# -*- coding: utf-8 -*-
# @File   : pointcloud_display.py
# @Date   : 2020/07/14
# @Author : PengLei
# @Mail   : penglei@chishine3d.com
# @bref   : This is a point cloud sample.
import ctypes
from openni import openni2
from openni import _openni2
import numpy as np
import cv2
import open3d
import device as cs
import struct

if __name__ == '__main__':
    openni2.initialize()
    dev = openni2.Device.open_any()    

    if dev.has_sensor(openni2.SENSOR_DEPTH):
        stream = dev.create_stream(openni2.SENSOR_DEPTH)

        # get and set video mode
        sensor_info = stream.get_sensor_info()
        curMode = None
        for i in range(len(sensor_info.videoModes)):
            mode = sensor_info.videoModes[i]
            print(mode.pixelFormat)
            print(mode.resolutionX)
            print(mode.resolutionY)
            print(mode.fps)

            if mode.resolutionX == 960:
                curMode = mode
         
        if curMode is not None:
           stream.set_video_mode(curMode)

        # start stream
        stream.start()

        # get Intrinsics
        intrinsics = stream.get_property(cs.CS_PROPERTY_STREAM_INTRINSICS, cs.Intrinsics)

        # get range of depth
        depthRange = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_DEPTH_RANGE, cs.DepthRange)
        print('DepthRang:', depthRange.min, depthRange.max)

        # set range of depth
        dr = cs.DepthRange(120, 500)
        stream.set_property(cs.CS_PROPERTY_STREAM_EXT_DEPTH_RANGE, dr)

        # set auto exposure off
        stream.set_property(_openni2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.c_uint32(0))
        # get exposure value
        exposure = stream.get_property(_openni2.ONI_STREAM_PROPERTY_EXPOSURE, ctypes.c_uint32)
        exposure = exposure.value + 100
        stream.set_property(_openni2.ONI_STREAM_PROPERTY_EXPOSURE, exposure)

        # set HDR mode
        stream.set_property(cs.CS_PROPERTY_STREAM_EXT_HDR_MODE, ctypes.c_uint32(2))
        # get current HDR mode
        hdr_mode = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_HDR_MODE, ctypes.c_ulong)
        print ('Get HDR_MODE:',hdr_mode.value)

        # get HdrScaleSetting
        hdrss = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_HDR_SCALE_SETTING, cs.HdrScaleSetting)
        print('Get HdrScaleSetting:\nhighReflectModeCount:%d\nhighReflectModeScale:%d\nlowReflectModeCount:%d\nlowReflectModeScale:%d\n'
              % (hdrss.highReflectModeCount, hdrss.highReflectModeScale, hdrss.lowReflectModeCount, hdrss.lowReflectModeScale))
        
        # get all params of HDR
        hdrexp = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_HDR_EXPOSURE, cs.HdrExposureSetting)        
        print ('Get HdrExposureSetting count =',hdrexp.count)
        for param in hdrexp.param:
            print ('exposure = {},gain = {}'.format(param.exposure, param.gain))

        # # get depth unit for real distance
        # dhsc = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_DEPTH_SCALE, ctypes.c_float)
        # print ('get depth scale:', dhsc.value)

        # get trigger mode 0-off 1-external trigger mode 2-software trigger mode
        trMode = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_TRIGGER_MODE, ctypes.c_int)
        print ('trigger mode:',trMode.value)

        # get min contrast value (remove where fringe contrast below this value)
        algcont = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_CONTRAST_MIN, ctypes.c_int)
        print ('algorithmContrast min:',algcont.value)

        # get frame time
        frTime = stream.get_property(cs.CS_PROPERTY_STREAM_EXT_FRAMETIME, ctypes.c_float)
        print ('get frame time:',frTime.value)

        # # show video
        # vis = open3d.visualization.Visualizer()
        # vis.create_window()        
        for count in range(100):
            frame = stream.read_frame()
            pc = cs.generatePointCloud(frame, intrinsics)

            print(count, end='-')
            
            if pc.any():
                print(pc.shape)
                # pct = open3d.geometry.PointCloud()
                # pct.points = open3d.utility.Vector3dVector(pc)
                # vis.clear_geometries()
                # vis.add_geometry(pct)
                # vis.run() 
            else:
                print()
        # vis.destroy_window()
        stream.stop()
    else:
        print("Device does not have depth sensor!")
    dev.close()
