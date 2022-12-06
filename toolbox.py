from functools import partial

import math3d as m3d
from scipy.optimize import least_squares
import numpy as np
import cv2


rcsv = partial(np.genfromtxt, delimiter=',')

# 特征点在bowl坐标系下的坐标
BOWL = np.array([
  [0.04000             , 0.008250  ,	0                     ],
  [0                   ,	0.008250  ,	0.04000               ],
  [-0.04000            ,	0.008250  ,	0                     ],
  [-0.0282842712474619 ,	0.008250  ,	-0.0282842712474619   ],
  [0.0                 ,	-0.02175  ,	-0.01500              ],
  [0.0106066017177982  ,	-0.02175  ,	-0.0106066017177982   ],
  [0.0106066017177982  ,	-0.02175  ,	0.0106066017177982    ],
  [-0.0106066017177982 ,	-0.02175  ,	0.0106066017177982    ],
])

# 远端锁定标定碗
LBWL = np.array([
  [ -0.0282842712474619,   0.0282842712474619,             0],
  [  0.0282842712474619,   0.0282842712474619,             0],
  [  0.0282842712474619,  -0.0282842712474619,             0],
  [               -0.04,                    0,             0],
  [-0.01767766952966369,  0.01767766952966369,         0.025],
  [               0.025,                    0,         0.025],
  [ 0.01767766952966369, -0.01767766952966369,         0.025],
  [-0.01767766952966369, -0.01767766952966369,         0.025],
])

def circle(img, pts, color=(0, 255, 0), raidus=8, lw=2):
  pts = np.array(pts)
  if pts.ndim == 1:
    pts = pts.reshape((-1, 2))

  for pt in pts:
    _pt = tuple(int(_) for _ in pt)

    cv2.circle(img, _pt, raidus, color, lw)

def cross_pt(p_raw):
  # 定义交点求解函数
  # 直线p1-p2和直线p3,p4的交点是p
  # 方法：l1p1+l2p2=l3p3+p4p4;
  #      l1+l2 = 1; l3+l4=1;
  # p1,p2,p3,p4已经实现进行了逆时针或顺时针排列
  p = p_raw.T.copy()
  p[:, 1] *= -1
  p[:, -1] *= -1
  A = np.vstack((p, np.array([[1, 0, 1, 0], [0, 1, 0, 1]])))
  b = np.array([0]*p.shape[0] + [1, 1])[:, None]
  X = np.linalg.inv(A) @ b

  return X
  
def assign_raw(pt_raw, pt_3d):
  pt_center = pt_raw.mean(0)
  l = pt_raw - pt_center
  theta = np.arctan2(l[:, 1], l[:, 0])
  pt_raw = pt_raw[np.argsort(theta)[::-1]]

  x_3D = cross_pt(pt_3d)
  x_2d = cross_pt(pt_raw)

  idx = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    [3, 2, 1, 0],
    [0, 3, 2, 1],
    [1, 0, 3, 2],
    [2, 1, 0, 3],
  ]
  err_tmp = np.inf
  for i in range(8):
    err = np.linalg.norm(x_3D - x_2d[idx[i]])
    if err < err_tmp:
      idx_best = i
      err_tmp = err
  return  pt_raw[idx[idx_best]]



# 使用Faugeras方法，计算相机内、外参初始值
def faugeras(pt_image, pt_bowl, camera_mode):
  # 输入：
  #   pt_image  : 标定碗上特征点在在图像坐标系(image)下的坐标, (N, 2)
  #   pt_bowl   : 标定碗在标定碗坐标系(bowl)下的坐标, (N, 3)
  #   camera_mode: 1 for LED, 0 for XRay
  # 输出：
  #   相机内参   ：A_image2cam
  #   相机外参   : p_cam2world
  N = len(pt_bowl) # 注册点个数
  AA1 = np.hstack((pt_bowl, np.ones((N, 1)))) # (8, 4)
  AA2 = - np.diag(pt_image[:, 0]) @ pt_bowl # (8, 3)
  AA3 = - np.diag(pt_image[:, 1]) @ pt_bowl # (8, 3)
  AA = np.vstack([
    np.hstack([             AA1, np.zeros((N, 4)),  AA2]),
    np.hstack([np.zeros((N, 4)),              AA1,  AA3]),
  ])
  BB = np.vstack((
    pt_image[:, 0:1],
    pt_image[:, 1:2]
  ))
  m_ = np.linalg.inv(AA.T @ AA) @ AA.T @ BB

  m34 = 1 / np.linalg.norm(m_[8:11])
  m = m34 * m_
  m1 = m[0:3]
  m14 = m[3, 0]
  m2 = m[4:7]
  m24 = m[7, 0]
  m3 = m[8:11]
  kx = np.linalg.norm(np.cross(m1.flat, m3.flat))
  ky = np.linalg.norm(np.cross(m2.flat, m3.flat))

  if camera_mode == 0:
    ky = -ky
  u = (m1.T @ m3).item()
  v = (m2.T @ m3).item()
  A_image2cam = np.array([
    [kx,  0,  u],
    [ 0, ky,  v],
    [ 0,  0,  1],
  ])
  R_cam2world = np.vstack([
    (m1 - u*m3).T / kx,
    (m2 - v*m3).T / ky,
    m3.T
  ])
  t_cam2world = np.array([
    (m14 - u*m34) / kx,
    (m24 - v*m34) / ky,
    m34,
  ])

  return A_image2cam, R_cam2world, t_cam2world

# 精确采用迭代法计算相机内、外参初始值
def reg_cam(pt_image, pt_bowl, camera_mode, A0=None):
  # 输入：
  #   pt_image  : 标定碗上特征点在在图像坐标系(image)下的坐标
  #   pt_bowl   : 标定碗在标定碗坐标系(bowl)下的坐标
  #   camera_mod: 相机类型，1为LED，0为X光
  #   A0        : 是否给定相机内参
  # 输出：
  #   相机内参   ：A_image2cam
  #   相机外参   : p_cam2world
  A0_image2cam, R0_cam2bowl, t0_cam2bowl = faugeras(pt_image, pt_bowl, camera_mode)
  if A0 is not None:
    A0_image2cam = A0

  T0_cam2bowl = m3d.Transform()
  T0_cam2bowl.set_orient(R0_cam2bowl)
  T0_cam2bowl.set_pos(t0_cam2bowl)
  # 使用Faugeras计算得出的待优化参数初始值
  X0 = np.array([
    A0_image2cam[0, 0], A0_image2cam[0, 2], A0_image2cam[1, 1], A0_image2cam[1, 2],
  ] + T0_cam2bowl.pose_vector.tolist())
  
  # 构造损失函数
  def err(x):
    assert len(x) == 10, 'Input param length must equals 10!'
    _A_image2cam = np.array([
      [x[0],    0, x[1]],
      [   0, x[2], x[3]],
      [   0,    0,    1]
    ])
    _T_cam2bowl = m3d.Transform(x[-6:])
    _pt_image_aug = _A_image2cam @ (_T_cam2bowl * pt_bowl.T)
    _pt_image = _pt_image_aug[:2, ...] / _pt_image_aug[2, ...]

    err0 = pt_image - _pt_image.T
    return err0.flatten()
    # return np.linalg.norm(err0, axis=-1)
  
  def err2(x):
    assert len(x) == 6, 'Input param length must equals 6 if A0 is given!'
    _T_cam2bowl = m3d.Transform(x)
    _pt_image_aug = A0 @ (_T_cam2bowl * pt_bowl.T)
    _pt_image = _pt_image_aug[:2, ...] / _pt_image_aug[2, ...]

    err0 = pt_image - _pt_image.T
    return err0.flatten()
    # return np.linalg.norm(err0, axis=-1)

  # 使用L-M法进行优化
  if A0 is None:
    res = least_squares(err, X0)
  else:
    res = least_squares(err2, T0_cam2bowl.pose_vector.tolist())
  return res.x

# 给定待优化参数，求解待优化参数在所有的图片下的重投影点
def x2repj(x, ps_base2ori, pts_bowl):
    # 优化参数必须包括：
    #   A_image2cam: 相机内参
    #   p_ori2tool: tcp_offset
    #   p_base2cam: 相机在基坐标系下的位置
    A_image2cam = np.array([
      [x[0],    0, x[1]],
      [   0, x[2], x[3]],
      [   0,    0,    1]
    ])
    p_ori2tool = x[4:10]
    p_base2cam = x[10:16]

    T_ori2tool = m3d.Transform(p_ori2tool)
    T_base2cam = m3d.Transform(p_base2cam)

    pts_image_rp = [] # 特征点在图像上的重投影
    for p_base2ori in ps_base2ori:
      T_base2ori = m3d.Transform(p_base2ori)
      pt_image_rp = A_image2cam @ (T_base2cam.inverse * T_base2ori * T_ori2tool * pts_bowl.T)
      pt_image_rp = pt_image_rp[:2, :] / pt_image_rp[2:, :]
      pt_image_rp = pt_image_rp.T
      pts_image_rp.append(pt_image_rp)
    
    return pts_image_rp

# 采用迭代法，计算相机内、外参，及标定碗的tcp_offset
def reg_sys(x0, pts_2d, ps_base2ori, pts_bowl):
  # 给定初始值
  # ps_base2ori : 所有图像中原始工具坐标的位置
  # pts_2d      : 所有图像中标志点的像素坐标
  # pts_bow     : 标定碗在tcp_offset坐标系下的坐标

  def err(x):
    # 优化参数包括：
    #   A_image2cam: 相机内参
    #   p_ori2tool: tcp_offset
    #   p_base2cam: 相机在基坐标系下的位置
    
    # 根据当前优化参数，计算图像中的重投影点
    pts_image_rp = x2repj(x, ps_base2ori, pts_bowl)
    pts_image_rp = np.array(pts_image_rp)
    
    pts_image = np.array(pts_2d)

    # 计算重投影点与原先点之间的误差
    diff = np.linalg.norm(pts_image - pts_image_rp, axis=-1)
    e = diff.sum(axis=0)
    # return  e
    return (pts_image - pts_image_rp).flatten()

  res = least_squares(err, x0)
  
  return  res.x