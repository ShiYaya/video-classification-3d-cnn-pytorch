
import os
import subprocess

path = '/home/yyshi/code/Action_Recognition/video-classification-3d-cnn-pytorch/msr-vtt_cliplen1.0/c2d_feats'

vids = os.listdir(path)
for vid in vids:
    vid_path = os.path.join(path, vid)
    new_vid = vid.split('.')[0] + '.npz'
    new_vid_path = os.path.join(path, new_vid)
    cmd = 'mv {} {}'.format(vid_path, new_vid_path)
    subprocess.call(cmd, shell=True)
    print()
