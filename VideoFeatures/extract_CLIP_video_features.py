#-*- coding: utf-8 -*-

# Code Reference: https://github.com/YoadTew/zero-shot-video-to-text/blob/35f8966684e2c47960f0e298e8aa35b4887f4cfe/run.py#L82
# Implemented by Yoonseok Heo at 230110
# Email: nlp.ysheo419@gmail.com

import clip, cv2, torch, os
import argparse, json
import os.path as osp
import numpy as np

from tqdm import tqdm
from PIL import Image
from datetime import datetime

def get_parser():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--dataset_name", type=str, required=True, choices=["MSR-VTT", "MSVD"])

    # Optional
    parser.add_argument("--dataset_path", type=str, default="../data")
    parser.add_argument("--output_directory_name", type=str, default="CLIP/features")
    parser.add_argument("--clip_pretrained_path", type=str, default="ViT-B/32")
    parser.add_argument("--filter_threshold", type=float, default=0.9)
    parser.add_argument("--max_sequence_frames", type=int, default=2000)

    return parser

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    datapath_dir_path = osp.join(args.dataset_path, args.dataset_name)
    modes = ["train", "valid", "test"]

    # Load a pretrained CLIP
    clip_model, clip_preprocess = clip.load(args.clip_pretrained_path)

    for mode in modes:
        print("  >> Start {} data processing...\n".format(mode))
        output_feature_path = osp.join(datapath_dir_path, mode, args.output_directory_name)
        if not osp.isdir(output_feature_path):
            os.makedirs(output_feature_path, exist_ok=True)
        with open(osp.join(output_feature_path, "../", "args.json"), "w") as f:
            args_dict = vars(args)
            x = datetime.now()
            args_dict["creation"] = str(x.year) + "_" + str(x.month) + "_" + str(x.day)
            json.dump(args_dict, f, indent="\t")
        print("  >> Features will be stored into {}".format(output_feature_path))

        cur_video_path_file = osp.join(datapath_dir_path, mode, "video_path.json")
        cur_video_info_file = osp.join(datapath_dir_path, mode, "video_info.json")
        if not osp.isfile(cur_video_path_file):
            assert FileNotFoundError
        if not osp.isfile(cur_video_info_file):
            assert FileNotFoundError
        with open(cur_video_path_file, "r") as f:
            video_path_list = json.load(f)
        with open(cur_video_info_file, "r") as f:
            video_info_dict = json.load(f)

        """
        errors = sanity_check_video_path(video_path_list)
        if errors > 0:
            print("Check the video path above...")
            exit(-1)
        print("  >> Pass sanity check...")
        """
        error_files = []
        for video_idx in tqdm(range(len(video_path_list)), desc="processing {}".format(mode), mininterval=0.01):
            video_path = video_path_list[video_idx]
            basename = osp.basename(video_path)
            vid, _ = osp.splitext(basename)
            video_info = video_info_dict[vid]

            sampled_frames = get_video_frames(video_path, video_info, clip_preprocess).to(device)
            if len(sampled_frames) == 0:
                error_files.append(video_path)
                continue        # data error
            with torch.no_grad():
                clip_model.eval()
                visual_feats = clip_model.encode_image(sampled_frames)
                frames_fts = torch.nn.functional.normalize(visual_feats, dim=-1)
                similarities = frames_fts @ frames_fts.T

                filtered_frame_feats, selected_frames_indices = filter_video(frames_fts, similarities)

            output_file_path = osp.join(output_feature_path, vid+".npy")
            np.save(output_file_path, filtered_frame_feats.detach().cpu().numpy())

        if len(error_files) != 0:
            print("  >> During {}, there are {} videos of which features cannot be extracted...".format(mode, len(error_files)))
            from pprint import pprint
            pprint(error_files)

            for epath in error_files:
                video_path_list.remove(epath)
                tt = osp.basename(epath)
                key = osp.splitext(tt)[0]
                del(video_info_dict[key])

            assert len(video_path_list) == len(video_info_dict)
            with open(cur_video_path_file, "w") as f:
                json.dump(video_path_list, f, indent="\t")
            with open(cur_video_info_file, "w") as f:
                json.dump(video_info_dict, f, indent="\t")
            print("  >> {} video information has been deleted in {} / {}".format(len(error_files), cur_video_path_file, cur_video_info_file))
            print("")


# Ref URL: https://github.com/YoadTew/zero-shot-video-to-text/blob/35f8966684e2c47960f0e298e8aa35b4887f4cfe/run.py
def filter_video(frame_fts, similarities, threshold=0.9):
    THRESHOLD=threshold
    groups = []
    curr_group = []
    for i in range(similarities.size(0)):
        if len(curr_group) == 0:
            curr_group.append(i)

        if i + 1 == similarities.size(0):
            if len(curr_group) >= 1:
                groups.append(curr_group)
            break

        if similarities[curr_group[0]][i + 1] > THRESHOLD:
            curr_group.append(i + 1)
        else:
            if len(curr_group) >= 1:
                groups.append(curr_group)
            curr_group = []

    result_features = []
    selected_indices = []
    if len(groups) >= 1:
        for i, group in enumerate(groups):
            result_features.append(frame_fts[group[0]])
            selected_indices.append(group[0])

    return torch.stack(result_features), selected_indices


# Ref URL: https://github.com/YoadTew/zero-shot-video-to-text/blob/35f8966684e2c47960f0e298e8aa35b4887f4cfe/run.py#L82
def get_video_frames(video_path, video_info, clip_preprocess):
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    sample_time = FPS // 3

    sampled_frames = []
    i=0
    while (cap.isOpened()):
        success, frame = cap.read()
        if success and i % sample_time == 0:
            # https://crmn.tistory.com/49
            # 파이썬에서 OpenCV를 사용해서 사진을 matplotlib 으로 화면에 출력하는 방법입니다.
            # 컬러 사진을 OpenCV에서는 BGR 순서로 저장하는데 matplotlib에서는 RGB 순서로 저장합니다.
            # 따라서 BGR을 RGB로 바꾸어 주어야만 사진이 제대로 표시됩니다.
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(converted)
            sampled_frames.append(pil_im)

        elif not success:
            break
        i += 1

    cap.release()
    if len(sampled_frames) == 0:
        sampled_feats = torch.tensor([])
    else:
        sampled_feats = torch.cat([clip_preprocess(x).unsqueeze(0) for x in sampled_frames])

    return sampled_feats

def sanity_check_video_path(video_path_list):
    cnt = 0
    for video_path in tqdm(video_path_list, desc="sanity check"):
        vvv = cv2.VideoCapture(video_path)
        if not vvv.isOpened():
            print("  >> Video is not opened! {}".format(video_path))
            cnt+= 1

    return cnt

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
