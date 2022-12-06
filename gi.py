from glob import glob

from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

import cv2
import json
import numpy as np
import itertools as it
import math3d as m3d

from toolbox import LBWL, reg_cam, circle, assign_raw

def video2gif(vid, output='output.gif'):
  videoClip = VideoFileClip(vid)
  print(videoClip.fps)

  videoClip = videoClip.fx( vfx.speedx, 2 )
  # videoClip.write_videofile('simple.mp4', audio=False)
  videoClip.write_gif(output, fps=videoClip.fps//4)


def video2frames(vid):
  cap = cv2.VideoCapture(vid)
  cnt = 0

  while True:
    ret, frame = cap.read()

    if not ret:
      break

    cv2.imshow('_', frame)

    if cv2.waitKey(1) == 27:
      break


def concatenate():
  """Concatenates several video files into one video file
  and save it to `output_path`. Note that extension (mp4, etc.) must be added to `output_path`
  `method` can be either 'compose' or 'reduce':
      `reduce`: Reduce the quality of the video to the lowest quality on the list of `video_clip_paths`.
      `compose`: type help(concatenate_videoclips) for the info"""
  # vids = glob('vids/*')
  vids = [

  ]
  method = 'compose'
  output = 'vids/cool.mp4'
  # create VideoFileClip object for each video file
  clips = [VideoFileClip(c) for c in vids]
  if method == "reduce":
      # calculate minimum width & height across all clips
      min_height = min([c.h for c in clips])
      min_width = min([c.w for c in clips])
      # resize the videos to the minimum
      clips = [c.resize(newsize=(min_width, min_height)) for c in clips]
      # concatenate the final video
      final_clip = concatenate_videoclips(clips)
  elif method == "compose":
      # concatenate the final video with the compose method provided by moviepy
      final_clip = concatenate_videoclips(clips, method="compose")
  
  # write the output video file
  final_clip.write_videofile(output)


T_plane2cam = np.array([
  [5193.85,        0,      633],  # 
  [      0,    -5197,      599],
  [      0,        0,        1],
])



if __name__ == '__main__':
  # P_CAM2BASE = [-0.0192333   0.04923981  0.98147314 -0.14786065 -0.8476393   1.88147681]

  with open('test\\test_kpts2.json', 'r') as f:
    kps = json.load(f)

  # # # 交换汶静标记点的x、y轴
  #   for kp in kps:
  #     for _k in ['keypoints', 'keypoints_12']:
  #       pts = np.array(kp[_k]).reshape((-1, 2))[:, ::-1]
  #       if _k == 'keypoints':
  #         kp[_k] = pts[4:].flatten().tolist() + pts[:4].flatten().tolist()
  #       else:
  #         kp[_k] = pts[6:].flatten().tolist() + pts[:6].flatten().tolist()
  
  # with open('test\\test_kpts2.json', 'w') as f:
  #   json.dump(kps, f)
  # print(kps)

  for kp in kps:
    image = f'test\\{kp["image_id"]}.bmp'
    img = cv2.imread(image)
    

    kpt_true = np.array(kp['keypoints']).reshape((-1, 2))
    kpt_true[:4, ...] = assign_raw(kpt_true[:4, ...], LBWL[:4, :2])
    kpt_true[4:, ...] = assign_raw(kpt_true[4:, ...], LBWL[4:, :2])

    kpt_tbd = np.array(kp['keypoints_12']).reshape((-1, 2))
    cnt = 0
    err_repj = np.inf
    kpt_repj = np.zeros_like(kpt_true)
    for _lidx in it.combinations(range(6), 4):
      for _sidx in it.combinations(range(6), 4):
        kpt_l = assign_raw(kpt_tbd[:6, :][_lidx, :], LBWL[:4, :2])
        kpt_s = assign_raw(kpt_tbd[6:, :][_sidx, :], LBWL[4:, :2])
        kpt_slct = np.vstack((kpt_l, kpt_s))

        p_cam2lbwl = reg_cam(kpt_slct, LBWL, 0, T_plane2cam)[-6:]
        kpt_cnt = T_plane2cam @ (m3d.Transform(p_cam2lbwl) * LBWL.T)
        kpt_cnt = (kpt_cnt[:2, :] / kpt_cnt[-1:, :]).T
        err_cnt = np.linalg.norm(kpt_cnt - kpt_slct)
        cnt += 1
        if err_cnt < err_repj or (cnt % 1000 == 0):
          err_repj = err_cnt
          kpt_repj = kpt_slct
          print(cnt, 'with err:', err_repj)

    circle(img, kpt_true[:4], raidus=12)
    circle(img, kpt_true[4:])
    circle(img, kpt_tbd[:6], (0, 0, 255), 24, lw=4)
    circle(img, kpt_tbd[6:], (0, 0, 255), 12, lw=4)
    circle(img, kpt_repj[:4], (0, 255, 255), 24)
    circle(img, kpt_repj[4:], (0, 255, 255), 12)

    cv2.imwrite(f'test\\{kp["image_id"]}_repj.bmp', img)
    print(f'-------- {kp["image_id"]} ------------')

    # cv2.imshow('_', cv2.resize(img, (512, 512)))
    # if cv2.waitKey(100) == 27:
    #   break





