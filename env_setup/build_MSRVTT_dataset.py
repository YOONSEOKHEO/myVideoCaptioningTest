#-*- coding: utf-8 -*-
# Implemented by Yoonseok Heo at 230109
# Email: nlp.ysheo419@gmail.com

import argparse, json, os
import os.path as osp
from collections import OrderedDict
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--train_val_abs_path", type=str, required=True, help="path should be absolute")
    parser.add_argument("--train_val_annotation_path", type=str, required=True, help="path should be absolute")
    parser.add_argument("--test_abs_path", type=str, required=True, help="path should be absolute")
    parser.add_argument("--test_annotation_path", type=str, required=True, help="path should be absolute")

    # Optional
    parser.add_argument("--output_path", type=str, default="../data/MSR-VTT")
    parser.add_argument("--video_path_file_name", type=str, default="video_path.json")
    parser.add_argument("--video_info_file_name", type=str, default="video_info.json")
    return parser


def main(args):
    train_dict, valid_dict, _, train_id, valid_id, _ = get_train_valid_test_info(annotation_path=args.train_val_annotation_path)
    train_valid_video_path = read_file_list(args.train_val_abs_path)
    train_video_list, valid_video_list = split_train_valid_video(train_id, train_valid_video_path)
    save_video_path(train_video_list, args.output_path, save_file_name=args.video_path_file_name, mode="train")
    save_video_path(valid_video_list, args.output_path, save_file_name=args.video_path_file_name, mode="valid")

    # get captions
    train_captions = split_captions(args.train_val_annotation_path, train_id, mode="train")
    valid_captions = split_captions(args.train_val_annotation_path, valid_id, mode="valid")

    for video_id, caption_list in train_captions.items():
        train_dict[video_id]["captions"] = caption_list
    for video_id, caption_list in valid_captions.items():
        valid_dict[video_id]["captions"] = caption_list

    save_video_path(train_dict, args.output_path, save_file_name=args.video_info_file_name, mode="train")
    save_video_path(valid_dict, args.output_path, save_file_name=args.video_info_file_name, mode="valid")

    _, _, test_dict, _, _, test_id = get_train_valid_test_info(args.test_annotation_path)
    test_video_list = read_file_list(args.test_abs_path)
    test_captions = split_captions(args.test_annotation_path, test_id)
    for video_id, caption_list in test_captions.items():
        test_dict[video_id]["captions"] = caption_list

    save_video_path(test_video_list, args.output_path, save_file_name=args.video_path_file_name, mode="test")
    save_video_path(test_dict, args.output_path, save_file_name=args.video_info_file_name, mode="test")

    return

def split_captions(annotation_path, id_list, mode="test"):
    with open(annotation_path, "r") as f:
        data = json.load(f)

    caption_dict = OrderedDict()
    sentences = data["sentences"]
    for caption_info in tqdm(sentences, desc="{}_split_captions".format(mode)):
        cur_video_id = caption_info["video_id"]
        if cur_video_id in id_list:
            if cur_video_id in caption_dict.keys():
                caption_dict[cur_video_id].append(caption_info["caption"])
            else:
                caption_dict[cur_video_id] = [caption_info["caption"]]

    return caption_dict


def split_train_valid_video(train_id, train_valid_video_path):
    train_path = []
    valid_path = []
    for video_path in train_valid_video_path:
        basename = osp.basename(video_path)
        file_name, ext = osp.splitext(basename)
        if file_name in train_id:
            train_path.append(video_path)
        else:
            valid_path.append(video_path)

    return train_path, valid_path

def get_train_valid_test_info(annotation_path):
    with open(annotation_path, "r") as f:
        data = json.load(f)

    train_dict = OrderedDict()
    valid_dict = OrderedDict()
    test_dict = OrderedDict()

    info_list = data["videos"]
    for info in info_list:
        if info["split"] == "train":
            train_dict[info["video_id"]] = {"video_info": info}
        elif info["split"] == "test":
            test_dict[info["video_id"]] = {"video_info": info}
        else:
            valid_dict[info["video_id"]] = {"video_info": info}

    return train_dict, valid_dict, test_dict, list(train_dict.keys()), list(valid_dict.keys()), list(test_dict.keys())



def save_video_path(file_list, output_dir_path, save_file_name, mode="train"):
    dir_path = osp.join(output_dir_path, mode)
    if not osp.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    #save_file_name = "video_path.txt"
    with open(osp.join(dir_path, save_file_name), "w") as f:
        json.dump(file_list, f, indent="\t")
        """
        for video_file_path in file_list:
            f.write(video_file_path)
            f.write("\n")
        """

    return

def read_file_list(directory_path):
    try:
        file_list = []
        for (root, directories, files) in os.walk(directory_path):
            files.sort()
            for file in files:
                file_path = osp.join(root, file)
                file_list.append(file_path)

    except FileNotFoundError:
        print("  >> Wrong file path: {}".format(directory_path))
        exit(-1)

    return file_list


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
