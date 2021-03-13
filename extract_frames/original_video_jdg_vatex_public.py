from __future__ import print_function, division
import os
import sys
import subprocess
import glob

import time
from multiprocessing.dummy import Pool as ThreadPool



def video_process(video_path):

  vid_name = video_path.split('/')[-1]
  dst_directory_path = os.path.join(dst_dir_path, vid_name[:-4])

  try:
    if os.path.exists(dst_directory_path):
      if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
        subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
        print('remove {}'.format(dst_directory_path))
        os.mkdir(dst_directory_path)

    else:
      os.mkdir(dst_directory_path)
  except:
    print(dst_directory_path)

  cmd = 'ffmpeg -i \"{}\" \"{}/image_%05d.jpg\"'.format(video_path, dst_directory_path)
  print(cmd)
  subprocess.call(cmd, shell=True)
  print('\n')

if __name__=="__main__":
  dir_path = "/home/Disk4T/zqzhang/shiyaya/dataset/VATEX/public_test_videos/*.mp4"
  dst_dir_path = "/home/Disk4T/zqzhang/shiyaya/dataset/VATEX/public_test_videos_frames"

  videos_path = glob.glob(dir_path)

  pool = ThreadPool(15)  # 创建10个容量的线程池并发执行
  pool.map(video_process, videos_path)  # pool.map同map用法
  pool.close()
  pool.join()
