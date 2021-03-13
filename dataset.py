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
import json


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
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_continuity_dataset(opt, video_path, sample_duration):
    dataset = []

    n_frames = len(os.listdir(video_path))

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }

    step = sample_duration
    for i in range(1, (n_frames - sample_duration + 1), step):
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(i, i + sample_duration))
        sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
        dataset.append(sample_i)

    return dataset

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

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'center_indice_path': begin_t,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }


    for center in indices:
        sample_i = copy.deepcopy(sample)
        sample_i['center_indice_path'] = os.path.join(video_path, 'image_{:05d}.jpg'.format(int(center)))
        sample_i['frame_indices'] = list(range(center-8, center+8))
        sample_i['segment'] = torch.IntTensor([center-8, center+7])
        dataset.append(sample_i)

    return dataset


class Video(data.Dataset):
    def __init__(self, opt, video_path, load_image_fn,
                 spatial_transform=None, temporal_transform=None,
                 sample_duration=16, get_loader=get_default_video_loader):

        self.load_image_fn = load_image_fn

        if opt.uniform_sampele:
            self.data = make_uniform_dataset(opt, video_path)
        else:
            self.data = make_continuity_dataset(opt, video_path, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        center_indice_path = self.data[index]['center_indice_path']

        if self.load_image_fn == None:
            return clip
        else:
            frame = self.load_image_fn(center_indice_path)
            return clip, frame

    def __len__(self):
        return len(self.data)
