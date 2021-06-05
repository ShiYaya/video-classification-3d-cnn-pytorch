videos_base_path = './zhihu_videos'
segment_videos_base_path = './process_zhihu_videos'

all_videos_list = []

import os
import subprocess
import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm

all_videos_list = os.listdir(videos_base_path)
for vid in tqdm(all_videos_list):
    vid_path = os.path.join(videos_base_path, vid)
    clip = VideoFileClip(vid_path)
    dur = clip.duration
    # dur = ffmpeg.probe(file_name)['format']['duration']
    
    for i in range(int(dur)//10):
        save_seg_path = os.path.join(segment_videos_base_path, vid[:-4]+'_seg_{}.mp4'.format(str(i).rjust(3,'0')))
        start = i*10
        last = 10 # 持续时间
        cmd = 'ffmpeg -i {} -ss {} -t {} -c copy {}'.format(vid_path, start, last, save_seg_path)
        subprocess.call(cmd, shell=True)