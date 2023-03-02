#-*- coding: utf-8 -*-
# Implemented by Yoonseok Heo at 230223
# Email: nlp.ysheo419@gmail.com

import argparse, json, os, glob, pickle
import os.path as osp
from collections import OrderedDict
from tqdm import tqdm

data_info_dict = {"caption": "raw-captions.pkl",
                  "train": "train_list.txt",
                  "valid": "val_list.txt",
                  "test": "test_list.txt"}
def get_parser():
    parser = argparse.ArgumentParser()

    # required
    # --mp4_video_dir_path /home/ubuntu/Datasets/MSVD/YouTubeClips_mp4
    parser.add_argument("--mp4_video_dir_path", type=str, required=True, help="path should be absolute")
    # --dataset_info_dir_path /home/ubuntu/Datasets/MSVD/msvd_data
    parser.add_argument("--dataset_info_dir_path", type=str, required=True,
                        help="absolute path containing split_info and captions")

    # Optional
    parser.add_argument("--ext", type=str, default=".mp4", help="video file extension")
    parser.add_argument("--output_path", type=str, default="../data/MSVD")
    parser.add_argument("--video_path_file_name", type=str, default="video_path.json")
    parser.add_argument("--video_info_file_name", type=str, default="video_info.json")

    return parser

def sanity_check(args):
    if not osp.isdir(args.mp4_video_dir_path):
        raise Exception("Check the mp4_video_dir_path again...")
    if not osp.isdir(args.dataset_info_dir_path):
        raise Exception("Check the dataset_info_dir_path again...")

    file_list = glob.glob(osp.join(args.dataset_info_dir_path, "*" + ".txt"))
    for file_name in file_list:
        file_name = osp.basename(file_name)
        flag = True
        for k, v in data_info_dict.items():
            if file_name == v:
                flag = False
        if flag is True:
            raise Exception("Check the file({}) into {}. It should be mapped to 'data_info_dict' defined as global variable above..."
                            .format(file_name, args.dataset_info_dir_path))

    if not osp.splitext(args.video_path_file_name)[1] == ".json":
        raise Exception("the extension of video_path_file_name should be 'json'")
    if not osp.splitext(args.video_info_file_name)[1] == ".json":
        raise Exception("the extension of video_path_file_name should be 'json'")

    return

def main(args):
    sanity_check(args)
    video_directory_path = args.mp4_video_dir_path
    info_dir_path = args.dataset_info_dir_path
    video_extension = args.ext
    output_dir_path = args.output_path
    if not osp.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)
    print("  >> Two files {} and {} will be located at {}".format(
        args.video_path_file_name, args.video_info_file_name, osp.abspath(output_dir_path)))

    # Step 1. get captions
    caption_dict = load_msvd_captions(info_dir_path)

    # Step 2. process train/valid/test data
    modes = ["train", "valid", "test"]      # mapped to the keys of 'data_info_dict'
    for mode in modes:
        video_path_list, video_info_dict = process_data(info_dir_path, mode, video_directory_path,
                                                        video_extension, caption_dict)

        save_files(video_path_list, video_info_dict, output_dir_path, mode,
                   args.video_path_file_name, args.video_info_file_name)

    return

# Step 3. Save video path and video_info
def save_files(video_path_list, video_info_dict, output_dir, mode, video_path_file_name, video_info_file_name):
    dir_path = osp.join(output_dir, mode)
    if not osp.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    video_path_path = osp.join(dir_path, video_path_file_name)
    video_info_file_path = osp.join(dir_path, video_info_file_name)

    with open(video_path_path, "w") as f:
        json.dump(video_path_list, f, indent="\t")
    with open(video_info_file_path, "w") as f:
        json.dump(video_info_dict, f, indent="\t")

    print("\n  >> # of {} data: {}".format(mode, len(video_path_list)))
    print("  >> {} and {} are stored into {}".format(video_path_file_name, video_info_file_name, dir_path))
    print("\n")

    return

# Step 2. process train/valid/test data
def process_data(info_dir_path, mode, video_directory_path, video_extension, caption_dict):
    file_path = osp.join(info_dir_path, data_info_dict[mode])
    with open(file_path, "r") as f:
        video_id_list = f.readlines()

    video_path_list = []
    video_info_dict = OrderedDict()
    for idx, vid in tqdm(enumerate(video_id_list), desc="process_{}_data".format(mode)):
        vid = vid.strip()
        vpath = osp.join(video_directory_path, vid+video_extension)
        cur_caption = caption_dict[vid]
        string_caption_list = [caption_list_to_string(cap) for cap in cur_caption]
        vinfo = get_video_info(vid, mode)

        video_path_list.append(vpath)
        video_info_dict[vid] = {"video_info": vinfo, "captions": string_caption_list}

    assert len(video_path_list) == len(video_info_dict)

    return video_path_list, video_info_dict

def get_video_info(vid, mode):
    info = {"video_id": vid, "id": vid, "split": mode}
    return info

def caption_list_to_string(caption_list):
    return " ".join(caption_list)

# Step 1. get captions
def load_msvd_captions(info_dir_path):
    file_path = osp.join(info_dir_path, data_info_dict["caption"])
    with open(file_path, "rb") as f:
        caption_dict = pickle.load(f)
    return caption_dict

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)