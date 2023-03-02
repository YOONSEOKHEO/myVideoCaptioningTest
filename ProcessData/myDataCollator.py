#-*- coding: utf-8 -*-
# Implemented by Yoonseok Heo at 230110
# Email: nlp.ysheo419@gmail.com

import torch
import numpy as np
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

class myDataFormat():
    def __init__(
            self,
            padded_video_feats: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            caption_features: BatchEncoding,
            all_caption_list: List,
            video_id_list: List,
            vid_tensor: torch.LongTensor = None,
            padded_decoder_input_ids: torch.FloatTensor = None,
            padded_decoder_attention_mask: torch.FloatTensor = None,
            padded_labels: Optional[torch.FloatTensor] = None,
            human_prompt_list: Optional[List] = None,
            decoder_input_length: Optional[List] = None,

    ):

        self.padded_video_feats = padded_video_feats
        self.attention_mask = attention_mask
        self.caption_features = caption_features
        self.all_caption_list = all_caption_list
        self.video_id_list= video_id_list
        self.vid_tensor = vid_tensor

        self.padded_decoder_input_ids = padded_decoder_input_ids
        self.padded_decoder_attention_mask = padded_decoder_attention_mask
        self.padded_labels = padded_labels

        self.human_prompt_list = human_prompt_list
        self.decoder_input_length = decoder_input_length

# Data collator(230215; myCollator + PTuneDataCollator 통합)
@dataclass
class PTuneDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    prompt_tokenizer: PreTrainedTokenizerBase

    pseudo_token: str = "[PROMPT]"
    template: Tuple[int, int, int] = (3, 3, 0)
    mode: str = "train"
    max_length: int = 25
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def get_p_tuned_input_format(self, human_prompt):
        pseudo_token_id = self.prompt_tokenizer.get_vocab()[self.pseudo_token]
        tok1 = [pseudo_token_id] * self.template[0]
        tok2 = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(human_prompt))
        tok3 = [pseudo_token_id] * self.template[1]
        input_tensor = torch.LongTensor(tok1+tok2+tok3)
        prompt_length = input_tensor.size(0)

        return input_tensor, prompt_length

    def build_prompting_input(self, ptune_input_tensor, prompt_length, caption_input_ids, caption_attn_mask):

        concat_ptune_input_tensor = torch.cat((ptune_input_tensor, caption_input_ids), dim=0)

        concat_ptune_input_tensor = concat_ptune_input_tensor.squeeze(0)

        # inference 시에도 feat_dict["caption"] 으로 [SOS] * batch_size 가 넘어 온다는 가정
        attn_mask_tensor = torch.cat((torch.ones(prompt_length), caption_attn_mask))

        # construct labels
        # prompt_length: -100, target_captions["attention_mask"][i]==0 ==> -100, 나머지는 input_ids 복붙
        # mask_value = torch.full([], -100, dtype=attn_mask_tensor.dtype).to(self.cuda_device)
        mask_value = torch.full([], -100, dtype=caption_input_ids.dtype)
        masked_labels = torch.where(caption_attn_mask.to(torch.bool), caption_input_ids, mask_value)
        # input_ids의 첫번째는 항상 SOS(==EOS)인데, 이것에 대응하는 labels를 -100으로 하기 위함.
        if caption_input_ids[0] == self.tokenizer.eos_token_id:
            masked_labels[0] = -100
        concat_labels = torch.cat((torch.full(size=(prompt_length,), fill_value=-100), masked_labels), dim=0)

        return concat_ptune_input_tensor, attn_mask_tensor, concat_labels


    def __call__(self, features, return_tensors=None):
        """
        :param features: a list of vocabs with visual_features and target captions
            - "video_id"
            - "video_feat"
            - "caption"
            - "human_prompt_list"
            - "all_caption_list"
        :param return_tensors:
        :return: batch - a dictionary of keys
        """
        feat_dict = {}
        keys = list(features[0].keys())  # visual, caption, all_caption_list, (human_prompt)
        num_patches = []
        for feat in features:
            for k, v in feat.items():
                if k in feat_dict:
                    feat_dict[k].append(v)
                else:
                    feat_dict[k] = [v]

                if k == "video_feat":
                    num_patches.append(v.size(0))

        # Step 1. 이미지 패딩
        #  - 최대 이미지 패치 개수를 뽑아서 padding 해주기
        #  - padding은 우선 가장 마지막 피쳐를 반복해서 붙여주는 걸로(efficient prompt 논문 참고)
        #  - 최종 output:
        #       * video features: (batch, #max_patches, embeddings)
        # TODO
        #  * attention_mask: (batch, #max_patches), 실제 patch만  GPT2에서 attention 하도록!
        #  * 근데 이게 안 맞는거 같음.. key.size() = batch, seq, seq ==> attn_mask?
        #  *

        attention_mask_list = []
        padded_video_feats = []
        max_patches = max(num_patches)
        for video_feat in feat_dict["video_feat"]:
            cur_size = video_feat.size(0)
            num_pads = max_patches - cur_size

            attn_mask = torch.zeros(cur_size)

            if num_pads != 0:
                num_pads = max_patches - cur_size
                pad_rows = video_feat[-1, :].unsqueeze(0).repeat(num_pads, 1)
                padded_video_feats.append(torch.cat([video_feat, pad_rows], dim=0))
                attn_mask = torch.cat([attn_mask, torch.ones(num_pads)], dim=-1)

            else:
                padded_video_feats.append(video_feat)

            attention_mask_list.append(attn_mask)

        padded_video_feats = torch.stack(padded_video_feats, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0).bool()
        assert padded_video_feats.size(0) == attention_mask.size(0)
        assert padded_video_feats.size(1) == attention_mask.size(1)

        # Step 2. 캡션 토큰화 및 패딩
        #  - batch 내 최대 길이 또는 max_sequence_length에 맞게 토큰화 하기
        #  - 한가지 확인할 부분은 GPT2의 스타트 토큰을 EOS로 주는게 맞는지 확인할 것
        #  - 최종 output:
        #       * labels: (batch, max_seq_len) , pad_token = -100
        #       * decoder_input_ids,

        caption_features = self.tokenizer(feat_dict["caption"], truncation=True, padding=True,
                                          max_length=self.max_length, return_tensors=self.return_tensors)

        # Step 3. Human Prompts --- in case of ViCap
        # prompt_features = None
        # if "human_prompts" in feat_dict:
        #     prompt_features = self.tokenizer(feat_dict["human_prompts"], truncation=True, padding=True,)
        # human_prompt + prompt_token 은 항상 prompt_tokenizer를 이용할 것!
        # build_prompt_encoder에 있는거 가져와서 넣기. 그리고 main함수에서 prompt_tokenizer 선언하기

        decoder_input_tensor = []
        decoder_attention_mask = []
        labels = []
        batch_size = padded_video_feats.size(0)
        decoder_input_length = []
        if self.mode == "test":
            # Set start-of-token(|ENDOFTEXT|) as the decoder_start_token
            sos_decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long) \
                                    * self.tokenizer.bos_token_id
            sos_decoder_attention_mask = torch.ones((batch_size, 1), dtype=torch.long)
            caption_input_ids = sos_decoder_input_ids
            caption_attn_mask = sos_decoder_attention_mask
        else:       # train/valid에는 항상 caption_features가 있을 것이다. 어떤 dataset 이 오더라도
            caption_input_ids = caption_features["input_ids"]
            caption_attn_mask = caption_features["attention_mask"]

        for i in range(batch_size):
            # Step 3-2. Tokenize human prompts(이전에 ptune_modelling.py 에서 get_p_tuned_input_format 함수)
            ptune_input_tensor, prompt_length = self.get_p_tuned_input_format(
                human_prompt = feat_dict["human_prompt_list"][i])

            concat_ptune_input_tensor, attn_mask_tensor, concat_labels = self.build_prompting_input(
                ptune_input_tensor, prompt_length, caption_input_ids[i], caption_attn_mask[i])

            decoder_input_length.append(len(concat_ptune_input_tensor))
            decoder_input_tensor.append(concat_ptune_input_tensor)
            decoder_attention_mask.append(attn_mask_tensor)
            labels.append(concat_labels)

        def do_left_padding(pad_size, pad_value, input_tensor):
            pad_tensor = torch.full(size=(pad_size,), fill_value=pad_value)
            concat_tensor = torch.cat([pad_tensor, input_tensor], dim=0)

            return concat_tensor


        # padded_queries : (batch, max_sequence_length)
        # prompt_tokens + human_prompt + prompt_tokens + target_caption + pad_tokens
        if self.mode != "test":
            padded_decoder_input_ids = pad_sequence(decoder_input_tensor, batch_first=True,
                                                    padding_value=self.prompt_tokenizer.pad_token_id).long()
            padded_decoder_attention_mask = pad_sequence(decoder_attention_mask, batch_first=True, padding_value=0).long()
            padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100).long()
        # inference 할때는 padding을 left에다가 해야 batch_generation이 가능
        else:
            assert len(decoder_input_tensor) == len(decoder_attention_mask)
            assert len(decoder_input_tensor) == len(labels)
            assert len(decoder_input_tensor) == len(decoder_input_length)
            max_len = max(decoder_input_length)
            left_pad = [max_len - len(cur_input) for cur_input in decoder_input_tensor]
            padded_decoder_input_ids = []
            padded_decoder_attention_mask = []
            padded_labels = []
            for idx, pad_size in enumerate(left_pad):
                if pad_size:
                    padded_decoder_input_ids.append(do_left_padding(pad_size,
                                                                    pad_value=self.prompt_tokenizer.pad_token_id,
                                                                    input_tensor=decoder_input_tensor[idx]))
                    padded_decoder_attention_mask.append(do_left_padding(pad_size, pad_value=0,
                                                                         input_tensor=decoder_attention_mask[idx]))
                    padded_labels.append(do_left_padding(pad_size, pad_value=-100, input_tensor=labels[idx]))

                else:
                    padded_decoder_input_ids.append(decoder_input_tensor[idx])
                    padded_decoder_attention_mask.append(decoder_attention_mask[idx])
                    padded_labels.append(labels[idx])

            padded_decoder_input_ids = torch.stack(padded_decoder_input_ids, dim=0)
            padded_decoder_attention_mask = torch.stack(padded_decoder_attention_mask, dim=0)
            padded_labels = torch.stack(padded_labels, dim=0)
            #print("PAUSE")

        assert padded_video_feats.size(0) == caption_features["input_ids"].size(0)
        assert padded_video_feats.size(0) == attention_mask.size(0)
        assert padded_video_feats.size(0) == padded_decoder_input_ids.size(0)
        assert padded_video_feats.size(0) == padded_decoder_attention_mask.size(0)
        assert padded_video_feats.size(0) == padded_labels.size(0)

        assert padded_decoder_input_ids.size(1) == padded_decoder_attention_mask.size(1)
        if not self.mode == "test":
            assert padded_decoder_input_ids.size(1) == padded_labels.size(1)

        return myDataFormat(padded_video_feats=padded_video_feats,
                            attention_mask=attention_mask,              # video feature에 대한 attention_mask
                            caption_features=caption_features if caption_features is not None else None,          # target caption의 input_ids, attention_mask
                            all_caption_list=feat_dict["all_caption_list"],
                            video_id_list=feat_dict["video_id"],

                            padded_decoder_input_ids= padded_decoder_input_ids,
                            padded_decoder_attention_mask=padded_decoder_attention_mask,
                            padded_labels = padded_labels,

                            human_prompt_list=feat_dict[
                                "human_prompt_list"] if "human_prompt_list" in feat_dict else None,
                            )





@dataclass
class myDataCollatorForSeq2Seq:

    tokenizer: PreTrainedTokenizerBase
    max_length: 25
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True

    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        """
        :param features: a list of vocabs with visual_features and target captions
            - "video_id"
            - "video_feat"
            - "caption"
            - "human_prompt_list"
            - "all_caption_list"
        :param return_tensors:
        :return: batch - a dictionary of keys
        """
        feat_dict = {}
        keys = list(features[0].keys())     # visual, caption, all_caption_list, (human_prompt)
        num_patches = []
        for feat in features:
            for k, v in feat.items():
                if k in feat_dict:
                    feat_dict[k].append(v)
                else:
                    feat_dict[k] = [v]

                if k == "video_feat":
                    num_patches.append(v.size(0))

        # Step 1. 이미지 패딩
        #  - 최대 이미지 패치 개수를 뽑아서 padding 해주기
        #  - padding은 우선 가장 마지막 피쳐를 반복해서 붙여주는 걸로(efficient prompt 논문 참고)
        #  - 최종 output:
        #       * video features: (batch, #max_patches, embeddings)
        # TODO
        #  * attention_mask: (batch, #max_patches), 실제 patch만  GPT2에서 attention 하도록!
        #  * 근데 이게 안 맞는거 같음.. key.size() = batch, seq, seq ==> attn_mask?
        #  *

        attention_mask_list = []
        padded_video_feats = []
        max_patches = max(num_patches)
        for video_feat in feat_dict["video_feat"]:
            cur_size = video_feat.size(0)
            num_pads = max_patches - cur_size

            attn_mask = torch.zeros(cur_size)

            if num_pads != 0:
                num_pads = max_patches - cur_size
                pad_rows = video_feat[-1, :].unsqueeze(0).repeat(num_pads, 1)
                padded_video_feats.append(torch.cat([video_feat, pad_rows], dim=0))
                attn_mask = torch.cat([attn_mask, torch.ones(num_pads)], dim=-1)

            else:
                padded_video_feats.append(video_feat)

            attention_mask_list.append(attn_mask)

        padded_video_feats = torch.stack(padded_video_feats, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0).bool()
        assert padded_video_feats.size(0) == attention_mask.size(0)
        assert padded_video_feats.size(1) == attention_mask.size(1)

        # Step 2. 캡션 토큰화 및 패딩
        #  - batch 내 최대 길이 또는 max_sequence_length에 맞게 토큰화 하기
        #  - 한가지 확인할 부분은 GPT2의 스타트 토큰을 EOS로 주는게 맞는지 확인할 것
        #  - 최종 output:
        #       * labels: (batch, max_seq_len) , pad_token = -100
        #       * decoder_input_ids,

        caption_features = self.tokenizer(feat_dict["caption"], truncation=True, padding=True,
                                          max_length=self.max_length, return_tensors=self.return_tensors)

        # Step 3. Human Prompts --- in case of ViCap
        # prompt_features = None
        # if "human_prompts" in feat_dict:
        #     prompt_features = self.tokenizer(feat_dict["human_prompts"], truncation=True, padding=True,)


        assert padded_video_feats.size(0) == caption_features["input_ids"].size(0)
        assert padded_video_feats.size(0) == attention_mask.size(0)

        return myDataFormat(padded_video_feats= padded_video_feats,
                            attention_mask= attention_mask,
                            caption_features= caption_features,
                            all_caption_list= feat_dict["all_caption_list"],
                            video_id_list= feat_dict["video_id"],
                            human_prompt_list = feat_dict["human_prompt_list"] if "human_prompt_list" in feat_dict else None,
                            )


        #return padded_video_feats, attention_mask, caption_features, feat_dict["all_caption_list"], feat_dict["video_id"]

# Data collator impleneted at 230216
@dataclass
class FTunePTuneDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    prompt_tokenizer: PreTrainedTokenizerBase

    prompt_flag: bool = False

    pseudo_token: str = "[PROMPT]"
    template: Tuple[int, int, int] = (3, 3, 0)
    mode: str = "train"
    max_length: int = 25
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"


    def get_p_tuned_input_format(self, human_prompt):
        pseudo_token_id = self.prompt_tokenizer.get_vocab()[self.pseudo_token]
        tok1 = [pseudo_token_id] * self.template[0]
        tok2 = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(human_prompt))
        tok3 = [pseudo_token_id] * self.template[1]
        input_tensor = torch.LongTensor(tok1+tok2+tok3)
        prompt_length = input_tensor.size(0)

        return input_tensor, prompt_length

    def build_prompting_input(self, ptune_input_tensor, prompt_length, caption_input_ids, caption_attn_mask):

        concat_ptune_input_tensor = torch.cat((ptune_input_tensor, caption_input_ids), dim=0)

        concat_ptune_input_tensor = concat_ptune_input_tensor.squeeze(0)

        # inference 시에도 feat_dict["caption"] 으로 [SOS] * batch_size 가 넘어 온다는 가정
        attn_mask_tensor = torch.cat((torch.ones(prompt_length), caption_attn_mask))

        # construct labels
        # prompt_length: -100, target_captions["attention_mask"][i]==0 ==> -100, 나머지는 input_ids 복붙
        # mask_value = torch.full([], -100, dtype=attn_mask_tensor.dtype).to(self.cuda_device)
        mask_value = torch.full([], -100, dtype=caption_input_ids.dtype)
        masked_labels = torch.where(caption_attn_mask.to(torch.bool), caption_input_ids, mask_value)
        # input_ids의 첫번째는 항상 SOS(==EOS)인데, 이것에 대응하는 labels를 -100으로 하기 위함.
        if caption_input_ids[0] == self.tokenizer.eos_token_id:
            masked_labels[0] = -100
        concat_labels = torch.cat((torch.full(size=(prompt_length,), fill_value=-100), masked_labels), dim=0)

        return concat_ptune_input_tensor, attn_mask_tensor, concat_labels


    def __call__(self, features, return_tensors=None):
        """
        :param features: a list of vocabs with visual_features and target captions
            - "video_id"
            - "vid" : video_id의 인덱스
            - "video_feat"
            - "caption"
            - "human_prompt_list"
            - "all_caption_list"
        :param return_tensors:
        :return: batch - a dictionary of keys
        """
        feat_dict = {}
        keys = list(features[0].keys())  # visual, caption, all_caption_list, (human_prompt)
        num_patches = []
        for feat in features:
            for k, v in feat.items():
                if k in feat_dict:
                    feat_dict[k].append(v)
                else:
                    feat_dict[k] = [v]

                if k == "video_feat":
                    num_patches.append(v.size(0))

        # Step 1. 이미지 패딩
        #  - 최대 이미지 패치 개수를 뽑아서 padding 해주기
        #  - padding은 우선 가장 마지막 피쳐를 반복해서 붙여주는 걸로(efficient prompt 논문 참고)
        #  - 최종 output:
        #       * video features: (batch, #max_patches, embeddings)
        # TODO
        #  * attention_mask: (batch, #max_patches), 실제 patch만  GPT2에서 attention 하도록!
        #  * 근데 이게 안 맞는거 같음.. key.size() = batch, seq, seq ==> attn_mask?
        #  *

        attention_mask_list = []
        padded_video_feats = []
        max_patches = max(num_patches)
        for video_feat in feat_dict["video_feat"]:
            cur_size = video_feat.size(0)
            num_pads = max_patches - cur_size

            attn_mask = torch.zeros(cur_size)

            if num_pads != 0:
                num_pads = max_patches - cur_size
                pad_rows = video_feat[-1, :].unsqueeze(0).repeat(num_pads, 1)
                padded_video_feats.append(torch.cat([video_feat, pad_rows], dim=0))
                attn_mask = torch.cat([attn_mask, torch.ones(num_pads)], dim=-1)

            else:
                padded_video_feats.append(video_feat)

            attention_mask_list.append(attn_mask)

        padded_video_feats = torch.stack(padded_video_feats, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0).bool()
        assert padded_video_feats.size(0) == attention_mask.size(0)
        assert padded_video_feats.size(1) == attention_mask.size(1)

        # Step 2. 캡션 토큰화 및 패딩
        #  - batch 내 최대 길이 또는 max_sequence_length에 맞게 토큰화 하기
        #  - 한가지 확인할 부분은 GPT2의 스타트 토큰을 EOS로 주는게 맞는지 확인할 것
        #  - 최종 output:
        #       * labels: (batch, max_seq_len) , pad_token = -100
        #       * decoder_input_ids,

        caption_features = self.tokenizer(feat_dict["caption"], truncation=True, padding=True,
                                          max_length=self.max_length, return_tensors=self.return_tensors)

        # Step 3. Human Prompts --- in case of ViCap
        # prompt_features = None
        # if "human_prompts" in feat_dict:
        #     prompt_features = self.tokenizer(feat_dict["human_prompts"], truncation=True, padding=True,)
        # human_prompt + prompt_token 은 항상 prompt_tokenizer를 이용할 것!
        # build_prompt_encoder에 있는거 가져와서 넣기. 그리고 main함수에서 prompt_tokenizer 선언하기

        decoder_input_tensor = []
        decoder_attention_mask = []
        labels = []
        batch_size = padded_video_feats.size(0)
        decoder_input_length = []
        if self.mode == "test":
            # Set start-of-token(|ENDOFTEXT|) as the decoder_start_token
            sos_decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long) \
                                    * self.tokenizer.bos_token_id
            sos_decoder_attention_mask = torch.ones((batch_size, 1), dtype=torch.long)
            caption_input_ids = sos_decoder_input_ids
            caption_attn_mask = sos_decoder_attention_mask
        else:       # train/valid에는 항상 caption_features가 있을 것이다. 어떤 dataset 이 오더라도
            caption_input_ids = caption_features["input_ids"]
            caption_attn_mask = caption_features["attention_mask"]
            # construct labels
            mask_value = torch.full([], -100, dtype=caption_input_ids.dtype)
            labels = torch.where(caption_attn_mask.to(torch.bool), caption_input_ids, mask_value)
            # 굳이 첫번째 labeL을 -100으로 설정하지 않아도, loss 계산과정에서 알아서 shift된다.
            # 하지만, prompt 경우에는 다르다. 왜냐하면, label 앞에 pseudo_token들이 붙기 때문에,
            # 아래 for문에서 masked_labels 만들때 mask_value를 앞에 prepend하기 전에 반드시 -100으로 처리되어야 함.

        # fine-tuning
        if self.prompt_flag is not True:
            return myDataFormat(padded_video_feats=padded_video_feats,
                                attention_mask=attention_mask,  # video feature에 대한 attention_mask
                                caption_features=caption_features if caption_features is not None else None,
                                # target caption의 input_ids, attention_mask
                                all_caption_list=feat_dict["all_caption_list"],
                                video_id_list=feat_dict["video_id"],
                                vid_tensor=torch.tensor(feat_dict["vid"], dtype=torch.int64),
                                padded_decoder_input_ids=caption_input_ids,
                                padded_decoder_attention_mask=caption_attn_mask,
                                padded_labels=labels,

                                human_prompt_list=None
                                )

        # p-tuning
        else:
            labels = []
            for i in range(batch_size):
                # Step 3-2. Tokenize human prompts(이전에 ptune_modelling.py 에서 get_p_tuned_input_format 함수)
                ptune_input_tensor, prompt_length = self.get_p_tuned_input_format(
                    human_prompt = feat_dict["human_prompt_list"][i])

                concat_ptune_input_tensor, attn_mask_tensor, concat_labels = self.build_prompting_input(
                    ptune_input_tensor, prompt_length, caption_input_ids[i], caption_attn_mask[i])

                decoder_input_length.append(len(concat_ptune_input_tensor))
                decoder_input_tensor.append(concat_ptune_input_tensor)
                decoder_attention_mask.append(attn_mask_tensor)
                labels.append(concat_labels)

            def do_left_padding(pad_size, pad_value, input_tensor):
                pad_tensor = torch.full(size=(pad_size,), fill_value=pad_value)
                concat_tensor = torch.cat([pad_tensor, input_tensor], dim=0)

                return concat_tensor


            # padded_queries : (batch, max_sequence_length)
            # prompt_tokens + human_prompt + prompt_tokens + target_caption + pad_tokens
            if self.mode != "test":
                padded_decoder_input_ids = pad_sequence(decoder_input_tensor, batch_first=True,
                                                        padding_value=self.prompt_tokenizer.pad_token_id).long()
                padded_decoder_attention_mask = pad_sequence(decoder_attention_mask, batch_first=True, padding_value=0).long()
                padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100).long()
            # inference 할때는 padding을 left에다가 해야 batch_generation이 가능
            else:
                assert len(decoder_input_tensor) == len(decoder_attention_mask)
                assert len(decoder_input_tensor) == len(labels)
                assert len(decoder_input_tensor) == len(decoder_input_length)
                max_len = max(decoder_input_length)
                left_pad = [max_len - len(cur_input) for cur_input in decoder_input_tensor]
                padded_decoder_input_ids = []
                padded_decoder_attention_mask = []
                padded_labels = []
                for idx, pad_size in enumerate(left_pad):
                    if pad_size:
                        padded_decoder_input_ids.append(do_left_padding(pad_size,
                                                                        pad_value=self.prompt_tokenizer.pad_token_id,
                                                                        input_tensor=decoder_input_tensor[idx]))
                        padded_decoder_attention_mask.append(do_left_padding(pad_size, pad_value=0,
                                                                             input_tensor=decoder_attention_mask[idx]))
                        padded_labels.append(do_left_padding(pad_size, pad_value=-100, input_tensor=labels[idx]))

                    else:
                        padded_decoder_input_ids.append(decoder_input_tensor[idx])
                        padded_decoder_attention_mask.append(decoder_attention_mask[idx])
                        padded_labels.append(labels[idx])

                padded_decoder_input_ids = torch.stack(padded_decoder_input_ids, dim=0)
                padded_decoder_attention_mask = torch.stack(padded_decoder_attention_mask, dim=0)
                padded_labels = torch.stack(padded_labels, dim=0)
                #print("PAUSE")

            assert padded_video_feats.size(0) == caption_features["input_ids"].size(0)
            assert padded_video_feats.size(0) == attention_mask.size(0)
            assert padded_video_feats.size(0) == padded_decoder_input_ids.size(0)
            assert padded_video_feats.size(0) == padded_decoder_attention_mask.size(0)
            assert padded_video_feats.size(0) == padded_labels.size(0)

            assert padded_decoder_input_ids.size(1) == padded_decoder_attention_mask.size(1)
            if not self.mode == "test":
                assert padded_decoder_input_ids.size(1) == padded_labels.size(1)

            return myDataFormat(padded_video_feats=padded_video_feats,
                                attention_mask=attention_mask,              # video feature에 대한 attention_mask
                                caption_features=caption_features if caption_features is not None else None,          # target caption의 input_ids, attention_mask
                                all_caption_list=feat_dict["all_caption_list"],
                                video_id_list=feat_dict["video_id"],
                                vid_tensor=torch.tensor(feat_dict["vid"], dtype=torch.int64),
                                padded_decoder_input_ids= padded_decoder_input_ids,
                                padded_decoder_attention_mask=padded_decoder_attention_mask,
                                padded_labels = padded_labels,

                                human_prompt_list=feat_dict[
                                    "human_prompt_list"] if "human_prompt_list" in feat_dict else None,
                                )
