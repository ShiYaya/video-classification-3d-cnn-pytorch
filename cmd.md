
CUDA_VISIBLE_DEVICES="1" python extract_feats_main_cliplen_vatex1.py --dataset vatex_trainval --clip_len 0.5


CUDA_VISIBLE_DEVICES="1" python extract_feats_main_cliplen_vatex2.py --dataset vatex_trainval --clip_len 0.5


CUDA_VISIBLE_DEVICES="1" python extract_feats_main_cliplen_vatex3.py --dataset vatex_trainval --clip_len 0.5


CUDA_VISIBLE_DEVICES="1" python extract_feats_main_cliplen_vatex4.py --dataset vatex_trainval --clip_len 0.5


CUDA_VISIBLE_DEVICES="1" python extract_feats_main_cliplen_vatex5.py --dataset vatex_trainval --clip_len 0.5


CUDA_VISIBLE_DEVICES="1" python extract_feats_main_cliplen_vatex2.py --dataset vatex_trainval --clip_len 0.5



CUDA_VISIBLE_DEVICES="0" python extract_feats_main_cliplen.py --dataset msrvtt --clip_len 0.5

