import cv2

from ChishineApi import CamChi






if __name__ == '__main__':
  cam_ip = '192.168.2.7'

  with CamChi(cam_ip) as cam:
    while True:
      pc, img = cam.grab()

      cv2.imshow('_', img)
      if cv2.waitKey(1) == 27:
        break