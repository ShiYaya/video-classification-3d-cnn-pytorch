import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import glob
import time

from opts import parse_opts
from model import generate_C3D_model
from mean import get_mean
from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding

def extract_feature(opt, video_dir, C3D_model):
    assert opt.mode in ['score', 'feature']

    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)
    load_image_fn = None
    data = Video(opt, video_dir, load_image_fn,
                 spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)

    c3d_features = []
    for i, clip in enumerate(data_loader):

        ## c3d feats
        clip = clip.to(opt.device)
        with torch.no_grad():
            c3d_outputs = C3D_model(clip)

        # 汇总
        c3d_features.append(c3d_outputs.cpu().data) #　torch.Size([8, 512, 14, 14])

    c3d_features = torch.cat(c3d_features, 0)  # c3d feature of one video


    return c3d_features.cpu().numpy()


def main(opt, C3D_model):
    all_videos_path = glob.glob(opt.video_root)
    # all_videos_path = sorted(all_videos_path, key=opt.video_sort_lambda)

    for video_path in all_videos_path:
        if os.path.exists(video_path):
            print(video_path)
            if opt.dataset == 'vatex':
                class_name = video_path.split('/')[-2]
                video_name = video_path.split('/')[-1][:-4]
                c3d_class_path = os.path.join(opt.output_c3d, class_name)
                if not os.path.exists(c3d_class_path):
                    os.makedirs(c3d_class_path)
                c3d_outfile = os.path.join(c3d_class_path, video_name + '.npy')

            else:
                video_name = video_path.split('/')[-1][:-4]
                c3d_outfile = os.path.join(opt.output_c3d, video_name + '.npy')


            # 暂时注释
            # if os.path.exists(c3d_outfile):
            #     continue

            start = time.time()
            # 截取帧
            subprocess.call('mkdir {}'.format(opt.tmp), shell=True)
            subprocess.call('ffmpeg -i "{}" {}/image_%05d.jpg'.format(video_path, opt.tmp),
                            shell=True)
            # 这里给 "video_path" 加引号，是为了处理 vatex 中类的文件名有空格、括号的情况


            if len(os.listdir(opt.tmp)) >= 1:
                # 提取特征
                # extract_feature, get numpy data
                c3d_features = extract_feature(opt, opt.tmp, C3D_model)

                # 保存特征到 npy 文件
                np.save(c3d_outfile, c3d_features)
            else:
                failed_path = './' + opt.dataset + '/failed_videos.txt'
                with open(failed_path, 'a+') as file:
                    file.write(video_path + '\n')

            # 删掉帧
            subprocess.call('rm -rf {}'.format(opt.tmp), shell=True)

            end = time.time()
            print("{}, use time:{}".format(video_name, end-start))
        else:
            print('{} does not exist'.format(video_name))

    if os.path.exists(opt.tmp):
        subprocess.call('rm -rf {}'.format(opt.tmp), shell=True)




if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.c3d_model_name, opt.c3d_model_depth)
    opt.mode = 'feature'
    opt.sample_size = 112
    opt.n_classes = 400


    ## C3D Model
    C3D_model = generate_C3D_model(opt)

    C3D_model.eval()
    if opt.verbose:
        print(C3D_model)

    ## FFMPEG
    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists(opt.tmp):
        subprocess.call('rm -rf {}'.format(opt.tmp), shell=True)

    main(opt, C3D_model)







