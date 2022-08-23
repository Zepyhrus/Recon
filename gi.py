from glob import glob

from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

import cv2



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



if __name__ == '__main__':
  vid = 'D:\\Seafile\\My Library\\Sugrical Robot\\media\\用户界面\\v1.webm'

  video2gif(vid)

  
  
  # concatenate()






