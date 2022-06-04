import cv2
import numpy as np


from ChishineApi import CamChi






if __name__ == '__main__':
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