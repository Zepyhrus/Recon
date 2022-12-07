from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

import cv2
import json
import numpy as np
import itertools as it
from math3d import Transform as Ts
from time import time

from toolbox import LBWL, reg_cam, circle, assign_raw

T_plane2cam = np.array([
  [5193.85,        0,      633],  # 
  [      0,    -5197,      599],
  [      0,        0,        1],
])

p_ori2lbwl = [0.00061, -0.00522, 0.186717, -1.58435, -0.00879, 0.00658]

if __name__ == '__main__':
  with open('test\\test_kpts2.json', 'r') as f:
    kps = json.load(f)
  
  ps = np.loadtxt('test\\ps_base2ori.txt')
  ps_base2ori = dict()
  for i in range(10):
    ps_base2ori[800+i] = ps[i]

  st = time()

  for kp in kps:
    if kp["image_id"] in [800, 802, 805, 808]: continue
    image = f'test\\{kp["image_id"]}.bmp'
    img = cv2.imread(image)

    # kpt_true = np.array(kp['keypoints']).reshape((-1, 2))
    # kpt_true[:4, ...] = assign_raw(kpt_true[:4, ...], LBWL[:4, :2])
    # kpt_true[4:, ...] = assign_raw(kpt_true[4:, ...], LBWL[4:, :2])

    kpt_tbd = np.array(kp['keypoints_12']).reshape((-1, 2))
    cnt = 0
    err_repj = np.inf
    kpt_repj = np.zeros((8, 2))
    for _lidx in it.combinations(range(6), 4):
      for _sidx in it.combinations(range(6), 4):
        kpt_l = assign_raw(kpt_tbd[:6][_lidx, :], LBWL[:4, :2])
        kpt_s = assign_raw(kpt_tbd[6:][_sidx, :], LBWL[4:, :2])
        kpt_slct = np.vstack((kpt_l, kpt_s))

        p_cam2lbwl = reg_cam(kpt_slct, LBWL, 0, T_plane2cam)
        kpt_cnt = T_plane2cam @ (Ts(p_cam2lbwl) * LBWL.T)
        kpt_cnt = (kpt_cnt[:2, :] / kpt_cnt[-1:, :]).T
        # err_cnt = np.linalg.norm(kpt_cnt - kpt_slct)
        err_cnt = np.abs(kpt_cnt - kpt_slct).max()
        cnt += 1
        if err_cnt < err_repj:
          err_repj = err_cnt
          kpt_repj = kpt_cnt # kpt_slct

          p_base2ori = ps_base2ori[kp['image_id']]
          T_base2cam = Ts(p_base2ori) * Ts(p_ori2lbwl) * Ts(p_cam2lbwl).inverse

          print(cnt, ':', err_repj, 'base2cam:', T_base2cam.pose_vector)
          print('cam2lbwl:', np.array(p_cam2lbwl))

    # circle(img, kpt_true[:4], raidus=12)
    # circle(img, kpt_true[4:])
    circle(img, kpt_tbd[:6], (0, 0, 255), 24, lw=4)
    circle(img, kpt_tbd[6:], (0, 0, 255), 12, lw=4)
    circle(img, kpt_repj[:4], (0, 255, 255), 24)
    circle(img, kpt_repj[4:], (0, 255, 255), 12)

    cv2.imwrite(f'test\\{kp["image_id"]}_repj.bmp', img)
    print(f'-------- {kp["image_id"]} ------------')

    # cv2.imshow('_', cv2.resize(img, (512, 512)))
    # if cv2.waitKey(100) == 27:
    #   break
  print(time() - st)





