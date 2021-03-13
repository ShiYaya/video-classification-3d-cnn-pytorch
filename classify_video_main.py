import h5py
import numpy as np
import glob
import torch
from torch import nn

import argparse
from model import generate_C3D_model


def prepare_trainval_video_id2name(opt):
    video_name2id = {}
    video_id2name = {}
    with open(opt.vatex_mapping, 'r') as file:
        data = file.readlines()
        for line in data:
            name, vid = line.strip().split(" ")
            vid = int(vid[3:]) - 1
            video_name2id[name] = vid
            video_id2name[vid] = name

    return video_id2name


def prepare_class_idx_mapping(opt):
    class_idx_mapping = {}
    with open(opt.kinetics400_classes_idx_map, 'r') as file:
        data = file.readlines()
        for idx, line in enumerate(data):
            cls_name = line.strip()
            class_idx_mapping[idx] = cls_name

    return class_idx_mapping


def general_classify(opt, c3d_fc_layer):
    video_id2name = prepare_trainval_video_id2name(opt)
    class_idx_mapping = prepare_class_idx_mapping(opt)

    video_features = h5py.File(opt.c3d_conv_feats_h5_path, 'r')
    c3d_conv_feats = video_features['motion_feats'][:]

    n_videos = len(c3d_conv_feats)
    assert n_videos == opt.n_videos
    with open(opt.pred_class_idx_path, 'w') as file:

        for idx in range(n_videos):
            video_idx = idx
            video_name = video_id2name[video_idx]
            conv_tensor = torch.from_numpy(c3d_conv_feats[idx]).to(opt.device)
            output = c3d_fc_layer(conv_tensor) # shape = (num_seg, n_cls)

            output = output.cpu().data.numpy()
            logit = np.sum(output, 0)
            top5_video_cls_idx = list(np.argsort(logit)[::-1][:5])
            top5_video_cls_name = [class_idx_mapping[cls_idx] for cls_idx in top5_video_cls_idx]

            print("{}\t{}\t{}\t{}\n".format(video_idx, video_name, top5_video_cls_name, top5_video_cls_idx))
            file.write("{}\t{}\t{}\t{}\n".format(video_idx, video_name, top5_video_cls_name, top5_video_cls_idx))
    print("All done !")



def vatex_public_test_classify(opt, c3d_fc_layer):
    class_idx_mapping = prepare_class_idx_mapping(opt)

    public_test_3d_npy_path = glob.glob(opt.public_test_3d_npy_path)
    n_videos = len(public_test_3d_npy_path)
    assert n_videos == opt.public_test_n_videos

    with open(opt.pred_class_idx_path, 'w') as file:
        for index in range(n_videos):
            video_name = public_test_3d_npy_path[index].split('/')[-1][:-4]  # video name

            c3d_conv_feats = np.load(public_test_3d_npy_path[index])
            conv_tensor = torch.from_numpy(c3d_conv_feats).to(opt.device)
            output = c3d_fc_layer(conv_tensor) # shape = (num_seg, n_cls)

            output = output.cpu().data.numpy()
            logit = np.sum(output, 0)
            top5_video_cls_idx = list(np.argsort(logit)[::-1][:5]) # np.argsort是按照升序，因此需要[::-1] 使其倒置
            top5_video_cls_name = [class_idx_mapping[cls_idx] for cls_idx in top5_video_cls_idx]

            print("{}\t{}\t{}\t{}\n".format(index, video_name, top5_video_cls_name, top5_video_cls_idx))
            file.write("{}\t{}\t{}\t{}\n".format(index, video_name, top5_video_cls_name, top5_video_cls_idx))

    print("All done !")


def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='vatex', type=str, help='which dataset to extract features')
    parser.add_argument('--video_root', default=None, type=str, help='Root path of input videos')
    parser.add_argument('--c3d_model_checkpoint', default='./pretrained_models/resnext-101-kinetics.pth', type=str, help='Model file path')
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size for c3d ')
    parser.add_argument('--n_threads', default=2, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--c3d_model_name', default='resnext', type=str, help='Currently only support resnet')
    parser.add_argument('--c3d_model_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--sample_duration', default=16, type=int,
                        help='the number frames of each segments')
    parser.add_argument('--num_segments', default=28, type=int, help='the number to sample frames and the num of segments to extract c3d features')
    parser.add_argument('--uniform_sampele', default=True, type=bool,
                        help='use uniform to sample center indices, if not, use continuity to sample')

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()

    opt.device = torch.device('cuda:'+str(opt.gpu_id) if torch.cuda.is_available() else 'cpu')

    return opt

if __name__=="__main__":

    opt = parse_opts()
    opt.arch = '{}-{}'.format(opt.c3d_model_name, opt.c3d_model_depth)
    opt.mode = 'score'
    opt.sample_size = 112
    opt.n_classes = 400

    if opt.dataset == 'vatex':
        opt.public_test = False
        # if Ture, to pred public test videos action
        # if False, to pred trainval videos action
        opt.kinetics400_classes_idx_map = './class_names_list'

    # C3D Model
    C3D_model = generate_C3D_model(opt)
    C3D_model.eval()
    if opt.verbose:
        print(C3D_model)

    # Get c3d fc layer
    c3d_fc_layer = C3D_model.fc

    if opt.public_test:
        opt.public_test_n_videos = 6000
        # 预测的action label 要写入的文件
        opt.pred_class_idx_path = "./vatex_public_test_pred_c3d_class.txt"
        opt.public_test_2d_npy_path = "/home/Disk4T/zqzhang/shiyaya/dataset/VATEX/test_inceptionresnetv2_feats/*.npy"
        opt.public_test_3d_npy_path = "/home/Disk4T/zqzhang/shiyaya/dataset/VATEX/test_resnext101_c3d_feats/*.npy"

        vatex_public_test_classify(opt, c3d_fc_layer)
    else:
        opt.n_videos = 28991
        # 预测的action label 要写入的文件
        opt.pred_class_idx_path = './vatex_trainval_pred_c3d_class.txt'
        base_path = "/home/Disk4T/zqzhang/shiyaya/code/video_captioning/shiyaya_video_caption_RecNet/"
        opt.c3d_conv_feats_h5_path = base_path + 'feats/vatex/vatex_frames_features(fc+c3d).h5'
        opt.vatex_mapping = base_path + "datasets/VATEX/vatex_mapping.txt"

        general_classify(opt, c3d_fc_layer)







