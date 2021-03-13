import argparse
import torch
import os

## batch_size=8, c3d memeory =8400M
# inception 1286
# resnet 1148
# resnet inception 1352


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='vatex', type=str, help='which dataset to extract features')
    parser.add_argument('--video_root', default=None, type=str, help='Root path of input videos')
    parser.add_argument('--c3d_model_checkpoint', default='./pretrained_models/resnext-101-kinetics.pth', type=str, help='Model file path')
    parser.add_argument('--output_c2d', default=None, type=str, help='Output file path')
    parser.add_argument('--output_c3d', default=None, type=str, help='Output file path')
    parser.add_argument('--mode', default='feature', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size for c3d ')
    parser.add_argument('--n_threads', default=2, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--c2d_model_name', default='inceptionresnetv2', type=str, help='inception_v4 | inception_v3 | resnet152 | inceptionresnetv2')
    parser.add_argument('--c3d_model_name', default='resnext', type=str, help='Currently only support resnet')
    parser.add_argument('--c3d_model_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--tmp', default=None, type=str, help='the dir to save ffmpeg images')

    parser.add_argument('--sample_duration', default=16, type=int,
                        help='the number frames of each segments')
    parser.add_argument('--num_segments', default=28, type=int, help='the number to sample frames and the num of segments to extract c3d features')
    parser.add_argument('--uniform_sampele', default=True, type=bool,
                        help='use uniform to sample center indices, if not, use continuity to sample')

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()

    if opt.dataset == 'msvd':
        opt.video_root = '/userhome/dataset/MSVD/Video-Description-with-Spatial-Temporal-Attention/youtube/*.avi'
        opt.video_sort_lambda = lambda x: int(x.split('/')[-1][3:-4])
    elif opt.dataset == 'msr-vtt':
        opt.video_root = '/home/Disk4T/zqzhang/shiyaya/dataset/MSR-VTT/train-video/*.mp4'
        opt.video_sort_lambda = lambda x: int(x.split('/')[-1][5:-4])
    elif opt.dataset == 'vatex':
        opt.video_root = '/home/Disk4T/zqzhang/shiyaya/dataset/VATEX/trainval_videos/**/*.mp4'


    opt.tmp = os.path.join(opt.dataset, 'tmp')

    opt.output_c3d = os.path.join('./', opt.dataset, 'c3d_feats')
    if not os.path.exists(opt.output_c3d):
        os.makedirs(opt.output_c3d)

    opt.output_c2d = os.path.join('./', opt.dataset, 'c2d_feats')
    if not os.path.exists(opt.output_c2d):
        os.makedirs(opt.output_c2d)

    opt.device = torch.device('cuda:'+str(opt.gpu_id) if torch.cuda.is_available() else 'cpu')


    return opt
