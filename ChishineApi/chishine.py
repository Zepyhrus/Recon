import time

import cv2
import open3d as o3d
import ChishineApi as cs
from ChishineApi.device import *
import numpy as np

from openni import openni2 as o2, _openni2 as _o2

def main():
  cam = cs.Camera(exposure=12000, trigger_mode=2)
  cam.start()

  # vis = o3d.visualization.Visualizer()
  # vis.create_window()

  frame_count = 0
  while True:
    # capture colored point cloud
    pc, color = cam.get_color_point_cloud(frame_count=frame_count)

    cv2.imshow('_', color)
    if cv2.waitKey(1) == 27:
      break

    # # detect charuco markers
    # cau3d.charuco3d_detect(pc/1000, 255-color, cau3d.generate_charuco_board(board_type=cau3d.BOARD_TYPE_A),
    #                        refine=True, verbose=cau3d.SHOW_RESULT, roi_type=cau3d.ROI_SQUARE)

    frame_count += 1
    print(f"frame_count = {frame_count}, {pc.shape}, {color.shape}")

  # vis.destroy_window()
  cam.stop()
  del cam


def test_rgb():
  o2.initialize()

  cam_ip = b'192.168.1.37'
  cam = o2.Device(cam_ip)
  
  # SET RGB STREAM PARAMS
  video_mode_rgb = _o2.OniVideoMode(pixelFormat=_o2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                    resolutionX=1920,
                                    resolutionY=1080,
                                    fps=0)
  stream_rgb = cam.create_stream(o2.SENSOR_COLOR)

  stream_rgb.set_video_mode(video_mode_rgb)
  stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, False)
  stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE, False)
  stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_EXPOSURE, 20000)
  stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_GAIN, 5)
  
  # # Get properties of stream
  # intrinsics_depth = stream_depth.get_property(CS_PROPERTY_STREAM_INTRINSICS, Intrinsics)
  # intrinsics_rgb = stream_rgb.get_property(CS_PROPERTY_STREAM_INTRINSICS, Intrinsics)
  # extrinsics = cam.get_property(CS_PROPERTY_DEVICE_EXTRINSICS, Extrinsics)

  stream_rgb.start()
  # =================================================================='

  cnt = 0
  ts = [time.time()]
  while cnt < 36000:
    frame = stream_rgb.read_frame()

    
    frame_data = np.array(frame.get_buffer_as_triplet())\
      .reshape([frame.height, frame.width, 3])[..., ::-1]

    cv2.imshow('_', frame_data)
    if cv2.waitKey(1) == 27:
      break
    
    cnt += 1
    ts.append(time.time())
    print(cnt, len(ts) / (ts[-1] - ts[0] + 1e-3), len(ts))
    if len(ts) >= 10:
      ts.pop(0)
  # ==================================================================
  stream_rgb.stop()
  cam.close()


def test_depth():
  o2.initialize()

  cam_ip = b'192.168.1.37'
  cam = o2.Device(cam_ip)

  # SET DEPTH STREAM PARAMS
  video_mode_depth = _o2.OniVideoMode(pixelFormat=_o2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM,
                                      resolutionX=960,
                                      resolutionY=600,
                                      fps=0)
  stream_depth = cam.create_stream(o2.SENSOR_DEPTH)
  
  stream_depth.set_video_mode(video_mode_depth)
  stream_depth.set_property(CS_PROPERTY_STREAM_EXT_TRIGGER_MODE, ctypes.c_uint32(2))
  stream_depth.set_property(CS_PROPERTY_STREAM_EXT_DEPTH_RANGE, DepthRange(250, 750))
  stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.c_uint32(0))
  stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_EXPOSURE, ctypes.c_uint32(5000))
  stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_GAIN, 1)

  depth_scale = 0.1
  # ==================================================================
  stream_depth.start()

  cnt = 0
  ts = [time.time()]
  while cnt < 3600:
    frame = stream_depth.read_frame()
    frame_data = np.array(frame.get_buffer_as_uint16()).reshape([frame.height, frame.width])

    np.save(f'data/depth{cnt}', frame_data)

    cv2.imshow('_', frame_data)
    if cv2.waitKey(1) == 27:
      break
    if cnt == 0:
      print(frame)

    cnt += 1
    ts.append(time.time())
    print(f'Count: {cnt}-{len(ts)}, FPS: {(10/(ts[-1]-ts[0])):.2f}, shape: {frame_data.shape}')
    if len(ts) >= 10:
      ts.pop(0)
  # ==================================================================
  stream_depth.stop()
  cam.close()


def test_pc():
  o2.initialize()

  cam_ip = b'192.168.1.37'
  cam = o2.Device(cam_ip)
  # ===================================================================
  # SET RGB STREAM PARAMS
  video_mode_rgb = _o2.OniVideoMode(pixelFormat=_o2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                    resolutionX=1920,
                                    resolutionY=1080,
                                    fps=0)
  stream_rgb = cam.create_stream(o2.SENSOR_COLOR)

  stream_rgb.set_video_mode(video_mode_rgb)
  stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, False)
  stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE, False)
  stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_EXPOSURE, 20000)
  stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_GAIN, 5)

  # -------------------------------------------------------------------
  # SET DEPTH STREAM PARAMS
  video_mode_depth = _o2.OniVideoMode(pixelFormat=_o2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM,
                                      resolutionX=960,
                                      resolutionY=600,
                                      fps=0)
  stream_depth = cam.create_stream(o2.SENSOR_DEPTH)
  
  stream_depth.set_video_mode(video_mode_depth)
  stream_depth.set_property(CS_PROPERTY_STREAM_EXT_TRIGGER_MODE, ctypes.c_uint32(2))
  stream_depth.set_property(CS_PROPERTY_STREAM_EXT_DEPTH_RANGE, DepthRange(250, 750))
  stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.c_uint32(0))
  stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_EXPOSURE, ctypes.c_uint32(5000))
  stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_GAIN, 1)


  # Get properties of stream
  intrinsics_depth = stream_depth.get_property(CS_PROPERTY_STREAM_INTRINSICS, Intrinsics)
  intrinsics_rgb = stream_rgb.get_property(CS_PROPERTY_STREAM_INTRINSICS, Intrinsics)
  extrinsics = cam.get_property(CS_PROPERTY_DEVICE_EXTRINSICS, Extrinsics)
  depth_scale = 0.1
  # ==================================================================
  # GET READY TO ROLL
  stream_rgb.start()
  stream_depth.start()

  cnt = 0
  ts = [time.time()]
  while cnt < 3600:
    frame_rgb = stream_rgb.read_frame()
    frame_depth = stream_depth.read_frame()

    img_rgb = np.array(frame_rgb.get_buffer_as_triplet()).reshape([frame_rgb.height, frame_rgb.width, 3])[..., ::-1]
    img_depth = np.array(frame_depth.get_buffer_as_uint16()).reshape([frame_depth.height, frame_depth.width])
    img_depth = img_depth.astype(np.float32)

    height, width = img_depth.shape
    fx = intrinsics_depth.fx * width / intrinsics_depth.width
    fy = intrinsics_depth.fy * height / intrinsics_depth.height
    cx = intrinsics_depth.cx * width / intrinsics_depth.width
    cy = intrinsics_depth.cy * height / intrinsics_depth.height

    zs = img_depth * depth_scale
    grid_mat = np.mgrid[0:height, 0:width]
    xs = (grid_mat[1, :, :] - cx) * zs / fx
    ys = (grid_mat[0, :, :] - cy) * zs / fy
    pc = np.stack((xs, ys, zs), axis=-1)

    rmat = np.asarray(extrinsics.rotation).reshape((3, 3))
    tvec = np.asarray(extrinsics.translation).reshape((3, 1))
    
    pc_in_rgb = pc @ rmat.T + tvec.T
    h_in_rgb = np.array([
      [intrinsics_rgb.fx, intrinsics_rgb.zero01, intrinsics_rgb.cx],
      [intrinsics_rgb.zeor10, intrinsics_rgb.fy, intrinsics_rgb.cy]
    ])
    h_in_rgb[0, :] *= img_rgb.shape[1] / img_depth.shape[1]
    h_in_rgb[1, :] *= img_rgb.shape[0] / img_depth.shape[0]

    uv = pc_in_rgb @ h_in_rgb.T / pc_in_rgb[:, :, 2, None]  # todo: 除零的情况怎么办？

    color = cv2.remap(img_rgb, uv[:, :, 0].astype(np.float32), uv[:, :, 1].astype(np.float32), interpolation=cv2.INTER_LINEAR,
                     borderValue=(0, 255, 0))

    cv2.imshow('__', color)
    if cv2.waitKey(1) == 27:
      break

    cnt += 1
    ts.append(time.time())
    print(f'Count: {cnt}-{len(ts)}, FPS: {(10/(ts[-1]-ts[0])):.2f}, shape: {pc.shape}, {color.shape}')
    if len(ts) >= 10:
      ts.pop(0)
  # ==================================================================
  stream_depth.stop()
  stream_rgb.stop()
  cam.close()



class CamChi:
  def __init__(self, cam_ip) -> None:
    o2.initialize()

    if type(cam_ip) is bytes:
      self.cam_ip = cam_ip
    elif type(cam_ip) is str:
      self.cam_ip = cam_ip.encode('utf-8')
    else:
      raise TypeError(f'cam type must be either string or bytes, instead of { type(cam_ip) }')
    
    self.cam = o2.Device(self.cam_ip)
    self.__cnt = 0
    self.__ts = []
    self.__ts_cache = 10
    # ===================================================================
    # SET RGB STREAM PARAMS
    video_mode_rgb = _o2.OniVideoMode(pixelFormat=_o2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                      resolutionX=1920,
                                      resolutionY=1080,
                                      fps=0)
    self.stream_rgb = self.cam.create_stream(o2.SENSOR_COLOR)
    self.stream_rgb.set_video_mode(video_mode_rgb)
    self.stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, False)
    self.stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE, False)
    self.stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_EXPOSURE, 10000)
    self.stream_rgb.set_property(_o2.ONI_STREAM_PROPERTY_GAIN, 5)

    # -------------------------------------------------------------------
    # SET DEPTH STREAM PARAMS
    video_mode_depth = _o2.OniVideoMode(pixelFormat=_o2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM,
                                        resolutionX=960,
                                        resolutionY=600,
                                        fps=0)
    self.stream_depth = self.cam.create_stream(o2.SENSOR_DEPTH)
    self.stream_depth.set_video_mode(video_mode_depth)
    self.stream_depth.set_property(CS_PROPERTY_STREAM_EXT_TRIGGER_MODE, ctypes.c_uint32(2))
    self.stream_depth.set_property(CS_PROPERTY_STREAM_EXT_DEPTH_RANGE, DepthRange(250, 750))
    self.stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.c_uint32(0))
    self.stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_EXPOSURE, ctypes.c_uint32(5000))
    self.stream_depth.set_property(_o2.ONI_STREAM_PROPERTY_GAIN, 1)


    # Get properties of stream
    self.intrinsics_depth = self.stream_depth.get_property(CS_PROPERTY_STREAM_INTRINSICS, Intrinsics)
    self.intrinsics_rgb = self.stream_rgb.get_property(CS_PROPERTY_STREAM_INTRINSICS, Intrinsics)
    self.extrinsics = self.cam.get_property(CS_PROPERTY_DEVICE_EXTRINSICS, Extrinsics)
    self.depth_scale = 0.1

  
  def __enter__(self):
    self.__ts.append(time.time())

    self.stream_depth.start()
    self.stream_rgb.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self):
    print(f'Cam {self.cam_ip} closed peacefully.')

    self.stream_depth.close()
    self.stream_rgb.close()
  
  def grab(self):
    frame_rgb = self.stream_rgb.read_frame()
    frame_depth = self.stream_depth.read_frame()

    img_rgb = np.array(frame_rgb.get_buffer_as_triplet()).reshape([frame_rgb.height, frame_rgb.width, 3])[..., ::-1]
    img_depth = np.array(frame_depth.get_buffer_as_uint16()).reshape([frame_depth.height, frame_depth.width])

    height, width = img_depth.shape
    fx = self.intrinsics_depth.fx * width / self.intrinsics_depth.width
    fy = self.intrinsics_depth.fy * height / self.intrinsics_depth.height
    cx = self.intrinsics_depth.cx * width / self.intrinsics_depth.width
    cy = self.intrinsics_depth.cy * height / self.intrinsics_depth.height

    zs = img_depth.astype(np.float32) * self.depth_scale
    grid_mat = np.mgrid[0:height, 0:width]
    xs = (grid_mat[1, :, :] - cx) * zs / fx
    ys = (grid_mat[0, :, :] - cy) * zs / fy
    pc = np.stack((xs, ys, zs), axis=-1)

    rmat = np.asarray(self.extrinsics.rotation).reshape((3, 3))
    tvec = np.asarray(self.extrinsics.translation).reshape((3, 1))
    
    pc_in_rgb = pc @ rmat.T + tvec.T
    h_in_rgb = np.array([
      [self.intrinsics_rgb.fx, self.intrinsics_rgb.zero01, self.intrinsics_rgb.cx],
      [self.intrinsics_rgb.zeor10, self.intrinsics_rgb.fy, self.intrinsics_rgb.cy]
    ])
    h_in_rgb[0, :] *= img_rgb.shape[1] / img_depth.shape[1]
    h_in_rgb[1, :] *= img_rgb.shape[0] / img_depth.shape[0]
    
    # TODO: 除零的情况怎么办？
    uv = pc_in_rgb @ h_in_rgb.T / pc_in_rgb[:, :, 2, None]
    uv = uv.astype(np.float32)

    color = cv2.remap(
      src=img_rgb,
      map1=uv[..., 0],
      map2=uv[..., 1],
      interpolation=cv2.INTER_LINEAR,
      borderValue=(0, 255, 0)
    )

    self.__cnt += 1
    self.__ts.append(time.time())
    if len(self.__ts)  >= self.__ts_cache:
      self.__ts.pop(0)
    
    return pc / 1000, color # 
  
  @property
  def fps(self):
    if len(self.__ts):
      return np.round(len(self.__ts) / (self.__ts[-1] - self.__ts[0] + 1e-3), 4)
    else:
      return 0
  
  @property
  def cnt(self):
    return self.__cnt


if __name__ == '__main__':
  cam_ip = '192.168.2.7'

  with CamChi(cam_ip) as cam:
    while True:
      pc, img = cam.grab()

      cv2.imshow('_', img)
      k = cv2.waitKey(1)

      print(cam.cnt, cam.fps)

      if k == 27:
        break
