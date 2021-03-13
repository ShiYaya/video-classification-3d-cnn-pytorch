# pip install tqdm
# pip install pillow
# pip install pretrainedmodels
# pip install nltk

# 没有使用extract_frames()--已经有了提取好的帧

import shutil
import subprocess
import threading, multiprocessing
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils
import time

C, H, W = 3, 224, 224
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)



def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.makedirs(dir_fc)
    print("save video feats to %s" % (dir_fc))
    video_list = os.listdir(params['video_path'])
    for video in tqdm(video_list):
        video_id = video

        if os.path.exists(os.path.join(dir_fc, video_id + '.npy')):
            continue

        dst = os.path.join(params["video_path"] , video_id)
        # extract_frames(video, dst)

        start = time.time()

        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))

        image_list = [image_list[int(sample)] for sample in samples]  # 平均采样28个
        images = torch.zeros((len(image_list), C, H, W))

        cpu_use_time = time.time() - start

        start = time.time()
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img

        load_image_time = time.time() - start


        start = time.time()
        with torch.no_grad():
            fc_feats = model(images.to(params['device'])).squeeze()
        img_feats = fc_feats.cpu().numpy()

        gpu_use_time = time.time() - start
        print('cpu:{}, load_image:{}, gpu:{}, all:{}'.format \
                  (cpu_use_time, load_image_time, gpu_use_time, \
                     cpu_use_time+load_image_time+gpu_use_time))

        # Save the inception features

        # 暂时
        # outfile = os.path.join(dir_fc, video_id + '.npy')
        # np.save(outfile, img_feats)

        # # cleanup
        # shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='tmp', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=28,
                        help='how many frames to sampler per video')
    # modified by yaya:
    # 由于已经提取了帧，因此这里，给定了帧的路径，而不是video的路径
    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='/userhome/dataset/MSR-VTT/train-video-frames', help='path to video dataset')

    parser.add_argument("--model", dest="model", type=str, default='inceptionresnetv2',
                        help='the CNN model you want to use to extract_feats')


    parser.add_argument('--gpu_id', type=int, default=0,
                    help='the gpu id to use')
    args = parser.parse_args()
    args.device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    args.output_dir = os.path.join(args.output_dir, args.model)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)


    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inceptionresnetv2':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))


    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)
    model = model.to(args.device)

    extract_feats(params, model, load_image_fn)


# 摘录来的
# class Identity(nn.Module):
#
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x