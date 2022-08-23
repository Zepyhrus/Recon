import cv2
import numpy as np
import open3d as o3d

from ChishineApi import CamChi

from pysuper4pcs import super4pcs_registration as reg_c # coarse registration
reg_f = o3d.pipelines.registration.registration_icp # fine registration


def cache_pc():
  cam_ip = '192.168.2.7'
  cnt = 0

  with CamChi(cam_ip) as cam:
    while True:
      pc, img = cam.grab()

      cv2.imshow('_', img)
      cnt += 1

      cv2.imwrite(f'data/{cnt}.png', img)
      np.save(f'data/{cnt}', pc)


      if cv2.waitKey(1) == 27 or cnt >= 100:
        break






if __name__ == '__main__':
  # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
  np_s = np.load('data/1.npy').reshape(-1, 3) * 1000
  np_t = np.load('data/2.npy').reshape(-1, 3) * 1000

  np_s = np_s[np.linalg.norm(np_s, axis=-1) > 1e-3]
  np_t = np_t[np.linalg.norm(np_t, axis=-1) > 1e-3]

  print(np_s.shape, np_t.shape)

  source = o3d.geometry.PointCloud()
  source.points = o3d.utility.Vector3dVector(np_s)

  target = o3d.geometry.PointCloud()
  target.points = o3d.utility.Vector3dVector(np_t)

  o3d.visualization.draw_geometries([source, target])


  # 粗配准
  score, T_coarse = reg_c(
    source=np.asarray(source.points),
    target=np.asarray(target.points),
    overlap=0.15,
    max_time_seconds=25,
    terminate_threshold=0.6,
    n_points=1000,
  )

  inv_T_coarse = np.linalg.inv(T_coarse)

  print(score, T_coarse)





  # # 精配准
  # reg_p2p = reg_f(
  #   source=target,
  #   target=source,
  #   max_correspondence_distance=0.5,
  #   init=inv_T_coarse,
  #   estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
  # )

  # print(reg_p2p.transformation)

  




