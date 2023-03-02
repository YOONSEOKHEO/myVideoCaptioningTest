#-*- coding: utf-8 -*-
# Implemented by Yoonseok Heo at 230110
# Email: nlp.ysheo419@gmail.com

import json, os, random, torch
import os.path as osp
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
random.seed(1)
# long_caption_flag일때
#   1. 같은 문장 제거 ==> 그냥 문장으로 만들어서 같은거 있으면 제외하기
#   2. 문장 뒤에 "." 넣기
#   3. 순서 셔플
#   4. num_caption_concat 만큼 문장들을 concat하기
#   5. 그것을 한 비디오에 대한 캡션으로 선택
#   6. (학습) 한 비디오에 캡션 1개
#       (검증, 평가): evalution_loss는 하나만 loss 계산은 무의미; 대신 concat_caption들 set 을 정답 set으로!

def get_caption_statistics(dataset):

    num_toks_list = []
    for idx, data in enumerate(dataset):
        caption = data["caption"]
        toks = caption.strip().split(" ")

        num_toks_list.append(len(toks))

    avg_size = sum(num_toks_list) / len(num_toks_list)
    max_size = max(num_toks_list)
    min_size = min(num_toks_list)

    num_toks_list = np.array(num_toks_list)
    print("  >> avg length of caption: {}".format(avg_size))
    print("  >> Max length of caption: {}".format(max_size))
    print("  >> Min length of caption: {}".format(min_size))
    print("  >> Standard Deviation: {}".format(np.std(num_toks_list)))
    return

class myDataset(Dataset):
    def __init__(self, data_dir_path, max_frames, samples=10, feature_type="CLIP",
                 mode="train", debug_flag=False, long_caption_flag=False, num_caption_concat=5,
                 cider_optimization_flag=False):

        assert mode.lower() in ["train", "valid", "test"]
        self.mode = mode.lower()
        self.max_frames = max_frames
        self.num_samples = samples
        self.long_caption_flag = long_caption_flag
        self.num_caption_concat = num_caption_concat
        self.cider_optimization_flag = cider_optimization_flag

        video_feature_dir = osp.join(data_dir_path, mode, feature_type, "features")
        feature_files = os.listdir(video_feature_dir)
        #self.video_feature_path = [osp.join(video_feature_dir, file_path) for file_path in feature_files]
        self.video_feature_path = {osp.splitext(file_path)[0]: osp.join(video_feature_dir, file_path) for file_path in feature_files}
        with open(osp.join(data_dir_path, mode, "video_info.json"), "r") as f:
            self.video_info = json.load(f)

        if self.long_caption_flag is True:
            self.dataset = self.build_long_captions_as_target()
            if debug_flag is True:
                get_caption_statistics(self.dataset)

        else:
            self.dataset = self.select_target_captions()

        if debug_flag is True:
            self.dataset = self.dataset[:32]

    def set_cider_optimization_flag(self):
        self.cider_optimization_flag = True
        self.dataset = self.select_target_captions()

    def get_all_target_captions_on_train_dataset(self):
        target_captions = [sample["caption"] for sample in self.dataset]

        return target_captions

    def build_long_captions_as_target(self):
        sampled_dataset = []
        for vid, value in self.video_info.items():
            cur_caps = value["captions"]
            num_caps = len(cur_caps)
            if self.mode == "train":
                long_caps = self.make_long_caption(caption_list = cur_caps, num_caption_concat = self.num_caption_concat)
                sampled_dataset.extend(
                    [{"video_id": vid, "caption": caption, "all_caption_list": []} for caption in long_caps])

            else:
                chosen_caps = random.sample(cur_caps, 1)        # chosen 1 caption for calculating loss
                sampled_dataset.extend([{
                    "video_id": vid, "caption": chosen_caps[0], "all_caption_list": cur_caps
                }])

        return sampled_dataset

    def make_long_caption(self, caption_list, num_caption_concat):
        targets = list(set(caption_list))
        # for idx in range(len(targets)):
        #     targets[idx] = targets[idx] + "."

        random.shuffle(targets)

        long_caption_list = []
        num_groups = len(targets) // num_caption_concat
        for idx in range(num_groups):
            long_cap = ". ".join(targets[idx * num_caption_concat: (idx + 1) * num_caption_concat])
            long_cap = long_cap + "."
            long_caption_list.append(long_cap)

        # remains = len(targets) - num_groups * num_caption_concat
        # if remains:
        #     long_cap = ". ".join(targets[num_groups * num_caption_concat:])
        #     long_cap = long_cap + "."
        #     long_caption_list.append(long_cap)

        return long_caption_list

    def select_target_captions(self):
        sampled_dataset  = []
        for vid, value in self.video_info.items():
            cur_caps = value["captions"]
            num_caps = len(cur_caps)
            if self.cider_optimization_flag is True:
                chosen_caps = random.sample(cur_caps, 1)  # chosen 1 caption for calculating loss
                sampled_dataset.extend([{
                    "video_id": vid, "caption": chosen_caps[0], "all_caption_list": cur_caps
                }])
            else:
                if self.mode == "train":
                    num_select = self.num_samples if self.num_samples <= num_caps else num_caps
                    chosen_caps = random.sample(cur_caps, num_select)
                    sampled_dataset.extend([{"video_id": vid, "caption": caption, "all_caption_list": []} for caption in chosen_caps])
                else:
                    chosen_caps = random.sample(cur_caps, 1)        # chosen 1 caption for calculating loss
                    sampled_dataset.extend([{
                        "video_id": vid, "caption": chosen_caps[0], "all_caption_list": cur_caps
                    }])
            #chosen_caps = random.sample(cur_caps, num_select)
            #sampled_dataset.extend([{"video_id": vid, "caption": caption} for caption in chosen_caps])

        return sampled_dataset

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):

        data = self.dataset[idx]
        sample = {}
        video_id = data["video_id"]
        target_caption = data["caption"]
        all_caption_list = data["all_caption_list"]

        video_feature_path = self.video_feature_path[video_id]
        visual_feature = torch.from_numpy(np.load(video_feature_path))
        if visual_feature.size(0) > self.max_frames:
            visual_feature = visual_feature[:self.max_frames, :]

        sample["video_id"] = video_id
        sample["video_feat"] = visual_feature
        sample["caption"] = target_caption
        sample["all_caption_list"] = all_caption_list

        return sample

# Debug
if __name__ == "__main__":
    dataset = myDataset(data_dir_path="data/MSR-VTT", feature_type="CLIP", mode="train")
    myDataLoader = DataLoader(dataset, batch_size=100, shuffle=False, )
    num_frames_train = []
    for batch in myDataLoader:
        #visual_features, captions = batch
        numFs = batch["numF"].detach().cpu().numpy()
        num_frames_train.extend(numFs)
        #print("PAUSE")

    dataset = myDataset(data_dir_path="data/MSR-VTT", feature_type="CLIP", mode="valid")
    myDataLoader = DataLoader(dataset, batch_size=100, shuffle=False)
    num_frames_valid = []
    for batch in myDataLoader:
        #visual_features, captions = batch
        numFs = batch["numF"].detach().cpu().numpy()
        num_frames_valid.extend(numFs)
        #print("PAUSE")

    dataset = myDataset(data_dir_path="data/MSR-VTT", feature_type="CLIP", mode="test")
    myDataLoader = DataLoader(dataset, batch_size=100, shuffle=False)
    num_frames_test = []
    for batch in myDataLoader:
        #visual_features, captions = batch
        numFs = batch["numF"].detach().cpu().numpy()
        num_frames_test.extend(numFs)
        #print("PAUSE")

    print("ALL_FINISH")