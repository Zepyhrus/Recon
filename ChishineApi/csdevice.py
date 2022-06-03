import ctypes
import numpy as np
from openni import openni2
from openni import _openni2
import cv2 as cv


class Camera:
  # camera 相关的宏与数据结构
  # **********************************************************************************************************************
  # Device property ID
  # **********************************************************************************************************************
  CS_PROPERTY_DEVICE_BASE = 0xD0000000
  CS_PROPERTY_DEVICE_IR_MODE = (CS_PROPERTY_DEVICE_BASE + 0x01)
  CS_PROPERTY_DEVICE_EXTRINSICS = (CS_PROPERTY_DEVICE_BASE + 0x02)

  # **********************************************************************************************************************
  # Extension property ID
  # **********************************************************************************************************************
  CS_PROPERTY_STREAM_BASE = 0xE0000000
  CS_PROPERTY_STREAM_INTRINSICS = (CS_PROPERTY_STREAM_BASE + 0x01)  # Intrinsics of depth camera or RGB camera
  CS_PROPERTY_STREAM_EXT_DEPTH_RANGE = (CS_PROPERTY_STREAM_BASE + 0x02)  # depth range of camera
  CS_PROPERTY_STREAM_EXT_HDR_MODE = (CS_PROPERTY_STREAM_BASE + 0x03)  # HDR mode
  CS_PROPERTY_STREAM_EXT_HDR_SCALE_SETTING = (CS_PROPERTY_STREAM_BASE + 0x04)  # setting of auto-HDR
  CS_PROPERTY_STREAM_EXT_HDR_EXPOSURE = (CS_PROPERTY_STREAM_BASE + 0x05)  # all params of HDR
  CS_PROPERTY_STREAM_EXT_DEPTH_SCALE = (CS_PROPERTY_STREAM_BASE + 0x06)  # depth unit for real distance
  CS_PROPERTY_STREAM_EXT_TRIGGER_MODE = (CS_PROPERTY_STREAM_BASE + 0x07)  # trigger mode
  CS_PROPERTY_STREAM_EXT_CONTRAST_MIN = (
      CS_PROPERTY_STREAM_BASE + 0x08)  # remove where fringe contrast below this value
  CS_PROPERTY_STREAM_EXT_FRAMETIME = (CS_PROPERTY_STREAM_BASE + 0x09)  # Frame time of depth camera

  # Distort of depth camera or RGB camera
  class Distort(ctypes.Structure):
    _fields_ = [("k1", ctypes.c_float),
                ("k2", ctypes.c_float),
                ("k3", ctypes.c_float),
                ("k4", ctypes.c_float),
                ("k5", ctypes.c_float)]

    def __repr__(self):
      return 'Distort(k1 = %r, k2 = %r, k3 = %r, k4 = %r, k5 = %r)' % (
        self.k1, self.k2, self.k3, self.k4, self.k5)

  # Intrinsics of depth camera or RGB camera
  class Intrinsics(ctypes.Structure):
    _fields_ = [("width", ctypes.c_short),
                ("height", ctypes.c_short),
                ("fx", ctypes.c_float),
                ("zero01", ctypes.c_float),
                ("cx", ctypes.c_float),
                ("zero10", ctypes.c_float),
                ("fy", ctypes.c_float),
                ("cy", ctypes.c_float),
                ("zero20", ctypes.c_float),
                ("zero21", ctypes.c_float),
                ("one22", ctypes.c_float)]

    def __repr__(self):
      return 'Intrinsics(width = %r, height = %r, fx = %r, fy = %r, cx = %r, cy = %r, zero01 = %r, zeor10 = %r, ' \
             'zeor20 = %r, zero21 = %r, one22 = %r)' % (
               self.width, self.height, self.fx, self.fy, self.cx, self.cy, self.zero01, self.zeor10, self.zeor20,
               self.zero21, self.one22)

  # Rotation and translation offrom depth camera to RGB camera
  class Extrinsics(ctypes.Structure):
    _fields_ = [("rotation", ctypes.c_float * 9),
                ("translation", ctypes.c_float * 3)]

    def __repr__(self):
      return 'Extrinsics(rotation = %r, translation = %r)' % (self.rotation, self.translation)

  # range of depth， value out of range will be set to zero
  class DepthRange(ctypes.Structure):
    _fields_ = [("min", ctypes.c_int),
                ("max", ctypes.c_int)]

    def __repr__(self):
      return 'DepthRange(min = %r, max = %r)' % (self.min, self.max)

  # exposure times and interstage scale of HDR
  class HdrScaleSetting(ctypes.Structure):
    _fields_ = [("highReflectModeCount", ctypes.c_uint),
                ("highReflectModeScale", ctypes.c_uint),
                ("lowReflectModeCount", ctypes.c_uint),
                ("lowReflectModeScale", ctypes.c_uint)]

    def __repr__(self):
      return 'HdrScaleSetting(highReflectModeCount = %r, highReflectModeScale = %r, lowReflectModeCount = %r, lowReflectModeScale = %r)' \
             % (
               self.highReflectModeCount, self.highReflectModeScale, self.lowReflectModeCount, self.lowReflectModeScale)

  # all exposure params of HDR
  class HdrExposureSetting(ctypes.Structure):
    class HdrExposureParam(ctypes.Structure):
      _pack_ = 1
      _fields_ = [("exposure", ctypes.c_uint),
                  ("gain", ctypes.c_ubyte)]

      def __repr__(self):
        return 'HdrExposureParam(exposure = %r, gain = %r)' % (self.exposure, self.gain)

    _pack_ = 1
    _fields_ = [("count", ctypes.c_ubyte),
                ("param", HdrExposureParam * 11)]

    def __repr__(self):
      return 'HdrExposureSetting(count = %r, param = %r)' % (self.count, self.param)

  def __init__(self, exposure=None, trigger_mode=0):
    # camera 的非静态成员
    self.dev = None
    self.stream_depth = None
    self.stream_rgb = None
    self.intrinsics_depth = None
    self.intrinsics_rgb = None
    self.extrinsics = None
    self.frame_data_depth = None
    self.frame_data_rgb = None
    self.depth_scale = None

    # ====================================================================================================
    # 初始化相机连接
    # ====================================================================================================
    openni2.initialize()
    self.dev = openni2.Device.open_any()

    if not self.dev.has_sensor(openni2.SENSOR_DEPTH):
      raise RuntimeError("Device does not have depth sensor!")
    if not self.dev.has_sensor(openni2.SENSOR_COLOR):
      raise RuntimeError("Device does not have color sensor!")

    self.stream_depth = self.dev.create_stream(openni2.SENSOR_DEPTH)
    self.stream_rgb = self.dev.create_stream(openni2.SENSOR_COLOR)

    # ====================================================================================================
    # 设置图像采集参数
    # ====================================================================================================
    video_mode_rgb = _openni2.OniVideoMode(pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                           resolutionX=1920, resolutionY=1080, fps=0)
    video_mode_depth = _openni2.OniVideoMode(pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM,
                                             resolutionX=960, resolutionY=600, fps=0)
    self.stream_depth.set_video_mode(video_mode_depth)
    self.stream_rgb.set_video_mode(video_mode_rgb)

    self.stream_depth.set_property(self.CS_PROPERTY_STREAM_EXT_TRIGGER_MODE, trigger_mode)

    # 如果给定了曝光值，则设置成固定曝光值，否则设为自动曝光
    if exposure:
      self.stream_depth.set_property(_openni2.ONI_STREAM_PROPERTY_EXPOSURE, ctypes.c_uint32(exposure))  # 设置成自动曝光
    else:
      self.stream_depth.set_property(_openni2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.c_uint32(1))  # 设置成自动曝光

    # ====================================================================================================
    # 读取相机内外参
    # ====================================================================================================
    self.intrinsics_depth = self.stream_depth.get_property(self.CS_PROPERTY_STREAM_INTRINSICS, self.Intrinsics)
    self.intrinsics_rgb = self.stream_rgb.get_property(self.CS_PROPERTY_STREAM_INTRINSICS, self.Intrinsics)
    self.extrinsics = self.dev.get_property(self.CS_PROPERTY_DEVICE_EXTRINSICS, self.Extrinsics)

  def __del__(self):
    if self.dev:
      self.stop()
      self.dev.close()
      print("\ncamera disconnected.")

  def start(self):
    self.stream_depth.start()
    self.stream_rgb.start()

  def stop(self):
    self.stream_depth.stop()
    self.stream_rgb.stop()

  def refresh_frame_depth(self, frame_count=1):
    frame = self.stream_depth.read_frame()
    self.frame_data_depth = np.array(frame.get_buffer_as_uint16()).reshape([frame.height, frame.width])
    if frame.videoMode.pixelFormat == _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM:
      self.depth_scale = 0.1
    elif frame.videoMode.pixelFormat == _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM:
      self.depth_scale = 1.0
    else:
      raise ValueError("Invalid pixelFormat!!!")

    if frame_count > 1:
      frame_data_ = np.zeros((frame_count, frame.height, frame.width), dtype=float)
      frame_data_[0, :, :] = self.frame_data_depth.copy()
      for i in range(frame_count - 1):
        frame = self.stream_depth.read_frame()
        frame_data_[i, :, :] = np.array(frame.get_buffer_as_uint16()).reshape([frame.height, frame.width])
      self.frame_data_depth = np.sum(frame_data_, axis=0) / \
                              np.maximum(np.sum(frame_data_ > 0, axis=0), 1)

  def refresh_frame_rgb(self):
    frame = self.stream_rgb.read_frame()
    self.frame_data_rgb = np.array(frame.get_buffer_as_triplet()).reshape([frame.height, frame.width, 3])

  def refresh_frame(self, frame_count=1):
    self.refresh_frame_depth(frame_count=frame_count)
    self.refresh_frame_rgb()

  def get_color_image(self):
    self.refresh_frame_rgb()
    return self.generate_color_image()

  def get_point_cloud(self, ordered=True, frame_count=1):
    self.refresh_frame_depth(frame_count=frame_count)
    return self.generate_point_cloud(ordered=ordered)

  def get_color_point_cloud(self, frame_count=1):
    self.refresh_frame(frame_count=frame_count)
    return self.generate_color_point_cloud()

  def generate_point_cloud(self, ordered=True):
    height = self.frame_data_depth.shape[0]
    width = self.frame_data_depth.shape[1]
    fx = self.intrinsics_depth.fx * width / self.intrinsics_depth.width
    fy = self.intrinsics_depth.fy * height / self.intrinsics_depth.height
    cx = self.intrinsics_depth.cx * width / self.intrinsics_depth.width
    cy = self.intrinsics_depth.cy * height / self.intrinsics_depth.height

    if ordered:
      zs = self.frame_data_depth * self.depth_scale
      grid_mat = np.mgrid[0:height, 0:width]
      xs = (grid_mat[1, :, :] - cx) * zs / fx
      ys = (grid_mat[0, :, :] - cy) * zs / fy
      pc = np.stack((xs, ys, zs), axis=-1)
    else:
      pc = []
      for v in range(height):
        for u in range(width):
          if self.frame_data_depth[v, u] > 0:
            z = self.frame_data_depth[v, u] * self.depth_scale
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pc.append([x, y, z])
      pc = np.array(pc)

    return pc

  def generate_color_image(self):
    R = self.frame_data_rgb[:, :, 0]
    G = self.frame_data_rgb[:, :, 1]
    B = self.frame_data_rgb[:, :, 2]
    image = np.transpose(np.array([B, G, R]), [1, 2, 0])
    return image

  def generate_color_point_cloud(self):
    pc = self.generate_point_cloud(ordered=True)
    rgb = self.generate_color_image()
    R = np.asarray(self.extrinsics.rotation).reshape((3, 3))
    t = np.asarray(self.extrinsics.translation).reshape((3, 1))
    pc_in_rgb_frame = pc @ R.T + t.T

    H_int_rgb = np.array([[self.intrinsics_rgb.fx, self.intrinsics_rgb.zero01, self.intrinsics_rgb.cx],
                          [self.intrinsics_rgb.zero10, self.intrinsics_rgb.fy, self.intrinsics_rgb.cy]])

    H_int_rgb[0, :] *= self.frame_data_rgb.shape[1] / self.frame_data_depth.shape[1]
    H_int_rgb[1, :] *= self.frame_data_rgb.shape[0] / self.frame_data_depth.shape[0]

    uv = pc_in_rgb_frame @ H_int_rgb.T / pc_in_rgb_frame[:, :, 2, None]  # todo: 除零的情况怎么办？

    color = cv.remap(rgb, uv[:, :, 0].astype(np.float32), uv[:, :, 1].astype(np.float32), interpolation=cv.INTER_LINEAR,
                     borderValue=(0, 255, 0))

    return pc_in_rgb_frame, color
