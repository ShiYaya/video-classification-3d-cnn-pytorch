## 说明
- 该份代码可以用来提取video的appearance feature(2D) and mototion feature(3D)
- opt 中设置了默认 2D: inceptionresnetv2; 3D:resnext C3D
- 2D 预训练模型不需要提前下载，3D 需要提前下载[pretrained model](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing)
并放在 pretrained_models 文件夹下，命名为resnext-101-kinetics.pth



# 特征提取
### appearence feature and motion feature
- 提取appearence feature and motion feature:    
  **单线程，for循环**： `video-classification-3d-cnn-pytorch/extract_feats_main.py`,    
  **多线程**： `video-classification-3d-cnn-pytorch/multi_thread_extract_feats_main.py`  
  该代码对一个video进行解帧，然后再调用 inceptionresnetv2 和 C3D的代码, 提取两个特征，然后把解的帧删除掉，再进行下一个video     
  
- 单独提取 appearence feature:    
  `video-classification-3d-cnn-pytorch/extract_fc_feats.py`    

- 单独 提取 C3D  feature：
   `video-classification-3d-cnn-pytorch/extract_c3d_feats_main.py`  


### 解帧 
- 专门解帧的一个代码：    
  vatex trainval 解帧: `video-classification-3d-cnn-pytorch/extract_frames/original_video_jdg_kinetics.py`  
  vatex public test 解帧：`video-classification-3d-cnn-pytorch/extract_frames/original_video_jdg_vatex_public.py`  
  
- 均匀采样，一个video 提取 28帧（采样方式保持与 c3d 代码中的采样方式一致)    
  (这里采样 28 帧主要是为 提取 object proposal 做准备 )  
  msvd/vtt: `video-classification-3d-cnn-pytorch/extract_frames/preprocess_frame.py`
  vatex: `video-classification-3d-cnn-pytorch/extract_frames/preprocess_frame_vatex.py`
  vatex_public_test: `video-classification-3d-cnn-pytorch/extract_frames/preprocess_frame_vatex_public.py`


## 按照 cliplen 来提取特征

<<<<<<< HEAD
python extract_feats_main_cliplen.py --dataset msr-vtt --clip_len 1.0
=======
python multi_thread_extract_feats_main_cliplen.py --dataset msr-vtt --clip_len 1.0
CUDA_VISIBLE_DEVICES="0" python extract_feats_main_cliplen_vatex1.py --dataset vatex_trainval --clip_len 1.0
CUDA_VISIBLE_DEVICES="0" python extract_feats_main_cliplen_vatex2.py --dataset vatex_trainval --clip_len 1.0
CUDA_VISIBLE_DEVICES="0" python extract_feats_main_cliplen_vatex3.py --dataset vatex_trainval --clip_len 1.0
>>>>>>> 3531ae1ba6b55bf3b59d6f9cd51f87c94dd83949

----------------------------
# 以下为源代码的readme

# Video Classification Using 3D ResNet
This is a pytorch code for video (action) classification using 3D ResNet trained by [this code](https://github.com/kenshohara/3D-ResNets-PyTorch).  
The 3D ResNet is trained on the Kinetics dataset, which includes 400 action classes.  
This code uses videos as inputs and outputs class names and predicted class scores for each 16 frames in the score mode.  
In the feature mode, this code outputs features of 512 dims (after global average pooling) for each 16 frames.  

**Torch (Lua) version of this code is available [here](https://github.com/kenshohara/video-classification-3d-cnn).**

## Requirements
* [PyTorch](http://pytorch.org/)
```
conda install pytorch torchvision cuda80 -c soumith
```
* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

## Preparation
* Download this code.
* Download the [pretrained model](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).  
  * ResNeXt-101 achieved the best performance in our experiments. (See [paper](https://arxiv.org/abs/1711.09577) in details.)

## Usage
Assume input video files are located in ```./videos```.

To calculate class scores for each 16 frames, use ```--mode score```.
```
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode score
```
To visualize the classification results, use ```generate_result_video/generate_result_video.py```.

To calculate video features for each 16 frames, use ```--mode feature```.
```
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode feature
```


## Citation
If you use this code, please cite the following:
```
@article{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  journal={arXiv preprint},
  volume={arXiv:1711.09577},
  year={2017},
}
```
