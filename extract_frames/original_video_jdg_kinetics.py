from __future__ import print_function, division
import os
import sys
import subprocess

import time
from multiprocessing.dummy import Pool as ThreadPool

dir_path = "/userhome/dataset/VATEX/trainval_videos/"
dst_dir_path = "/userhome/dataset/VATEX/original_trainval_frames/"


def class_process(class_name):

  class_path = os.path.join(dir_path, class_name)
  if not os.path.isdir(class_path):
    return

  dst_class_path = os.path.join(dst_dir_path, class_name)
  if not os.path.exists(dst_class_path):
    os.mkdir(dst_class_path)

  for file_name in os.listdir(class_path):
    if '.mp4' not in file_name:
      continue
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)
    try:
      if os.path.exists(dst_directory_path):
        if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
          subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
          print('remove {}'.format(dst_directory_path))
          os.mkdir(dst_directory_path)
        else:
          continue
      else:
        os.mkdir(dst_directory_path)
    except:
      print(dst_directory_path)
      continue
    cmd = 'ffmpeg -i \"{}\" \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')

if __name__=="__main__":
  dir_path = "/userhome/dataset/VATEX/trainval_videos/"
  dst_dir_path = "/userhome/dataset/VATEX/original_trainval_frames/"


  class_names = []
  for class_name in os.listdir(dir_path):
    class_names.append(class_name)
    print(class_name)


  pool = ThreadPool(15)  # 创建10个容量的线程池并发执行
  pool.map(class_process, class_names)  # pool.map同map用法
  pool.close()
  pool.join()
