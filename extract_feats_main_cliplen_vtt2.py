import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import glob
import time
from tqdm import tqdm

from opts_by_cliplen import parse_opts
from model import generate_C3D_model, generate_C2D_model
from mean import get_mean
from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding

from multiprocessing.dummy import Pool as ThreadPool

from moviepy.editor import VideoFileClip
import math


def extract_feature(opt, video_dir, C3D_model, load_image_fn, C2D_model, c2d_shape, duration):
    assert opt.mode in ['score', 'feature']
    C, H, W = c2d_shape

    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)

    opt.num_segments = int(duration/opt.clip_len)
    data = Video(opt, video_dir, load_image_fn,
                 spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=0, pin_memory=True)

    c3d_features = []
    c2d_features = []
    for i, (clip, frames_npy_data) in enumerate(data_loader):

        ## c3d feats
        clip = clip.to(opt.device)
        with torch.no_grad():
            c3d_outputs = C3D_model(clip)

        frames = frames_npy_data.to(opt.device)
        with torch.no_grad():
            c2d_outputs = C2D_model(frames).squeeze()
            if len(c2d_outputs.shape) == 1:
                c2d_outputs = c2d_outputs.unsqueeze(0)


        # 汇总
        c3d_features.append(c3d_outputs.cpu().data)
        c2d_features.append(c2d_outputs.cpu().data)
   
    try:
        c3d_features = torch.cat(c3d_features)  # c3d feature of one video
        c2d_features = torch.cat(c2d_features)  # c3d feature of one video
    except:
        return None, None


    return c3d_features.cpu().numpy(), c2d_features.cpu().numpy()


def main(video_path):
    outputs = []

    if os.path.exists(video_path):
        print(video_path)
        if opt.dataset == 'vatex_trainval':
            class_name = video_path.split('/')[-2]
            video_name = video_path.split('/')[-1][:-4]
        else:
            video_name = video_path.split('/')[-1][:-4]


        c3d_outfile = os.path.join(opt.output_c3d, video_name + '.npz')
        c2d_outfile = os.path.join(opt.output_c2d, video_name + '.npz')


        # 暂时注释
        if os.path.exists(c3d_outfile) and os.path.exists(c2d_outfile):
            return

        current_video_tmp = os.path.join(opt.tmp, video_name)
        if os.path.exists(current_video_tmp):
            subprocess.call('rm -rf {}'.format(current_video_tmp), shell=True)
        if not os.path.exists(current_video_tmp):
            os.makedirs(current_video_tmp)


        start = time.time()
        # 截取帧
        
        subprocess.call('mkdir {}'.format(current_video_tmp), shell=True)
        try:
            subprocess.call('ffmpeg -i "{}" {}/image_%05d.jpg'.format(video_path, current_video_tmp),
                            shell=True,
                            timeout=5)
        except:
            return
        # 这里给 "video_path" 加引号，是为了处理 vatex 中类的文件名有空格、括号的情况

        if len(os.listdir(current_video_tmp)) >= 1:
            # 提取特征
            # extract_feature, get numpy data
            duration = VideoFileClip(video_path).duration
            c3d_features, c2d_features = extract_feature(opt, current_video_tmp, C3D_model, load_image_fn, C2D_model, c2d_shape, duration)
            try:
                np.savez(c3d_outfile, features=c3d_features)
                np.savez(c2d_outfile, features=c2d_features)
            except:
                failed_path = './' + opt.dataset+'_cliplen{}'.format(opt.clip_len) + '/failed_videos.txt'
                with open(failed_path, 'a+') as file:
                    file.write(video_path + '\n')
        else:
            failed_path = './' + opt.dataset+'_cliplen{}'.format(opt.clip_len) + '/failed_videos.txt'
            with open(failed_path, 'a+') as file:
                file.write(video_path + '\n')

        # 删掉帧
        subprocess.call('rm -rf {}'.format(current_video_tmp), shell=True)

        end = time.time()
        print("{}, use time:{}".format(video_name, end-start))
    else:
        print('{} does not exist'.format(video_path))


if __name__ == '__main__':


    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.c3d_model_name, opt.c3d_model_depth)
    opt.sample_size = 112
    opt.n_classes = 400


    ## C3D Model
    C3D_model = generate_C3D_model(opt)

    C3D_model.eval()
    if opt.verbose:
        print(C3D_model)

    ## C2D Model
    load_image_fn, C2D_model, c2d_shape = generate_C2D_model(opt)
    C2D_model.eval()
    if opt.verbose:
        print(C2D_model)

    ## FFMPEG
    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'



    all_videos_path = glob.glob(opt.video_root)
    for vid_path in tqdm(all_videos_path[5000:]):
        main(vid_path)
