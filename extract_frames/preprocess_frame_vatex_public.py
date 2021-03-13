from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from PIL import Image
import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import copy
import glob
import numpy as np
import subprocess
import opts
from multiprocessing.dummy import Pool as ThreadPool
import tqdm



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        # image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)




def make_uniform_dataset(opt, video_path):
    dataset = []

    n_frames = len(os.listdir(video_path))

    last_frame = sorted(glob.glob(video_path + '/*.jpg'), key=lambda x: int(x[-9:-4]))[-1]
    while n_frames < opt.num_segments:
        n_frames += 1
        pad_frame = last_frame[:-9] + '%05d.jpg' % n_frames
        cmd = 'cp -r "{}" "{}"'.format(last_frame, pad_frame)
        print(cmd)
        subprocess.call('cp -r "{}" "{}"'.format(last_frame, pad_frame), shell=True)

    indices = np.linspace(9, n_frames+1 - 7, opt.num_segments, endpoint=False, dtype=int)
    # indices 存放的是 center indices of each segments
    # and the select [left 8 frames + this indice + right 7 frames]
    # note: the frames id from 1(not from 0), so, the start of np.linspace is 9


    center_indice_path = list(indices)

    sample = {
        'video': video_path,
        'center_indice_path': center_indice_path,
    }

    return sample



def Video(video_path):

    global opt
    global spatial_transform
    global loader

    vid = video_path.split('/')[-1]
    # class_name = video_path.split('/')[-2]
    data = make_uniform_dataset(opt, video_path)
    path = data['video']

    frame_indices = data['center_indice_path']
    # clip = loader(path, frame_indices)
    # if spatial_transform is not None:
    #     clip = [spatial_transform(img) for img in clip]
    #
    # if not os.path.exists(os.path.join(opt.save_path, vid)):
    #     os.makedirs(os.path.join(opt.save_path, vid))

    for i, image in enumerate(frame_indices):
        image_path = os.path.join(video_path, 'image_%05d.jpg'%frame_indices[i])
        assert os.path.exists(image_path)
        # print(image_path)
        target_image_path = os.path.join(opt.save_path, vid, 'image_%05d.jpg'%i)
        if not os.path.exists(os.path.join(opt.save_path, vid)):
            os.makedirs(os.path.join(opt.save_path, vid))
        cmd = 'cp -r "{}" "{}"'.format(image_path, target_image_path)
        subprocess.call(cmd, shell=True)
        print(cmd)


    # print(vid)



if __name__ == '__main__':

    opt = opts.parse_opts()
    opt.sample_size = 112
    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size)])
    loader = get_default_video_loader()

    base_dir = "/home/Disk4T/zqzhang/shiyaya/dataset/VATEX/public_test_videos_frames/*"
    videos_dir = glob.glob(base_dir)

    opt.save_path = "/home/Disk4T/zqzhang/shiyaya/dataset/VATEX/28frame-vatex_public/"
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    # for video_path in videos_dir:
    #     Video(video_path)


    pool = ThreadPool(6)  # 创建4个容量的线程池并发执行
    pool.map(Video, videos_dir)  # pool.map同map用法
    pool.close()
    pool.join()

