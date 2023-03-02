#-*- coding: utf-8 -*-
# Implemented by Yoonseok Heo at 230112
# Email: nlp.ysheo419@gmail.com
# Code Reference URL
#  - VisualGPT: https://github.com/Vision-CAIR/VisualGPT
#  - P-tuning: https://github.com/THUDM/P-tuning
"""
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
"""
import copy, os
import torch
import math
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
#from models.containers import Module, ModuleList
from torch.nn import Module, ModuleList
from transformers import AutoModelWithLMHead, AutoConfig, GPT2PreTrainedModel, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn.parameter import Parameter
from myModels.load_pretrained_gpt2 import load_weight
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.configuration_utils import PretrainedConfig

# root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # ~/myVideoCaptioning
#
# #pretrained_gpt2_path = os.path.join(root_dir_path, "pretrained_gpt2", "gpt2-pytorch_model.bin")
# pretrained_gpt2_path = os.path.join(root_dir_path, "pretrained_gpt2", "gpt2-medium-pytorch_model.bin")
# #
# original_state_dict = torch.load(pretrained_gpt2_path, map_location='cpu' if not torch.cuda.is_available() else None)

class MyGPT2Decoder(Module):
    def __init__(self, args, tokenizer, encoder, bos_idx, n_layer=12, tau=0):
        super(MyGPT2Decoder, self).__init__()
        self.args = args
        self.encoder = encoder
        self.bos_idx = bos_idx
        self.tokenizer = tokenizer

        config = AutoConfig.from_pretrained(args.model_name)
        config.add_cross_attention = True                       # 나중에 gpt2 generate 할때 쓸려고
        config.n_layer = n_layer

        #pretrained_decoder = AutoModelWithLMHead.from_pretrained(args.model_name)
        decoder = myGPT2LMHeadModel(config, padding_idx=self.tokenizer.pad_token_id, tau=tau)
        # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        decoder.config.pad_token_id = decoder.config.eos_token_id
        decoder.generation_config.pad_token_id = decoder.generation_config.eos_token_id

        root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/myVideoCaptioning
        if args.model_name == "gpt2":
            pretrained_gpt2_path = os.path.join(root_dir_path, "pretrained_gpt2", "gpt2-pytorch_model.bin")
        elif args.model_name == "gpt2-medium":
            pretrained_gpt2_path = os.path.join(root_dir_path, "pretrained_gpt2", "gpt2-medium-pytorch_model.bin")
        else:
            raise ValueError("model_name should be one of two: gpt2, gpt2-medium")

        original_state_dict = torch.load(pretrained_gpt2_path,
                                         map_location='cpu' if not torch.cuda.is_available() else None)

        decoder = load_weight(decoder, original_state_dict)

        self.decoder = decoder
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, padded_video_feats, attention_mask, target_captions,
                inputs_embeds = None, decoder_inputs_attention_mask = None,
                cider_optimization_flag=False,
                generate_option=False, generate_type="beam", **gen_kwargs):
        """
        :param padded_video_feats:
        :param attention_mask:      # video_feat에 대한 attention_mask로서 pad부분 = 0
        :param target_captions:
        :param inputs_embeds: batch, max_sequence_length, hidden_size
            # Token_embedding for P-tuning
        :param decoder_inputs_attention_mask: batch, max_sequence_length
            # 0: masking; attention mask for decoder inputs; only activated when inputs_embeds is not None
        :param generate_option:
        :param generate_type:
        :param bos_token_id:
        :return:
        """
        # enc_output: batch_size, #args.tfm_layers, #max_frames, feat_dim(512)
        # masked_enc: batch_size, 1, 1, max_frames ==> decoder 때문에
        enc_output, masked_enc = self.encoder(padded_video_feats, attention_mask)

        # Only activated for p-tuning; inputs_embeds and decoder_inputs_attention_mask must exist at the same time
        if inputs_embeds is not None:
            decoder_input_ids = None
            decoder_attention_mask = decoder_inputs_attention_mask
        else:
            decoder_input_ids = target_captions["input_ids"]
            decoder_attention_mask = target_captions["attention_mask"]

        if generate_option is True:
            if generate_type.lower() == "beam":
                # self.decoder.generate(encoder_output=enc_output, mask_encoder=masked_enc,
                #                       max_length=25, num_beams=3, early_stopping=True)
                beam_hyper_params = {"max_length": 25,
                                     "num_beams": 3,
                                     "early_stopping": True}
                batch_size = enc_output.size(0)

                if cider_optimization_flag is True:
                    from transformers import (
                        LogitsProcessorList, MinLengthLogitsProcessor,
                        StoppingCriteriaList, MaxLengthCriteria,
                        BeamSearchScorer, NoRepeatNGramLogitsProcessor,
                    )
                    from transformers.generation.utils import GenerationMixin

                    logits_processor = LogitsProcessorList(
                        [
                            MinLengthLogitsProcessor(5, eos_token_id=self.decoder.config.eos_token_id),
                            NoRepeatNGramLogitsProcessor(gen_kwargs["no_repeat_ngram_size"])
                        ]
                    )
                    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=gen_kwargs["max_length"])])

                    # instantiate beam scorer
                    beam_scorer = BeamSearchScorer(
                        batch_size=batch_size,
                        num_beams=gen_kwargs["num_beams"],
                        device=self.decoder.device,
                        do_early_stopping=gen_kwargs["early_stopping"],
                        num_beam_hyps_to_keep=gen_kwargs["num_return_sequences"],
                    )

                    model_kwargs = {}
                    model_kwargs["encoder_hidden_states"] = enc_output
                    model_kwargs["encoder_attention_mask"] = masked_enc
                    model_kwargs["attention_mask"] = decoder_attention_mask
                    model_kwargs["output_attentions"] = False
                    model_kwargs["output_hidden_states"] = False
                    model_kwargs["use_cache"] = True
                    input_ids, model_kwargs = self.decoder._expand_inputs_for_generation(
                        expand_size = gen_kwargs["num_beams"],
                        is_encoder_decoder = True,  # Yoon-edited
                        input_ids = decoder_input_ids,
                        ** model_kwargs,
                    )
                    with torch.enable_grad():
                        generated_ids = self.decoder.beam_search(
                            input_ids,
                            beam_scorer,
                            logits_processor=logits_processor,
                            stopping_criteria=stopping_criteria,
                            pad_token_id=self.decoder.generation_config.pad_token_id,
                            eos_token_id=self.decoder.generation_config.eos_token_id,
                            output_scores = True,
                            return_dict_in_generate = True,
                            **model_kwargs
                        )

                else:
                    generated_ids = self.decoder.generate(
                        input_ids               = decoder_input_ids,
                        attention_mask          = decoder_attention_mask,
                        encoder_hidden_states   = enc_output,
                        encoder_attention_mask  = masked_enc,
                        #**beam_hyper_params,
                        **gen_kwargs
                    )

                return generated_ids

        else:
            # dec_output, past = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            #                                 encoder_output=enc_output, mask_encoder=masked_enc)

            decoder_output = self.decoder(
                input_ids       = decoder_input_ids,
                inputs_embeds   = inputs_embeds,
                attention_mask  = decoder_attention_mask,
                encoder_hidden_states = enc_output,
                encoder_attention_mask = masked_enc,
            )
            return decoder_output
            #return dec_output, past


class myGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, padding_idx=47932, tau=0):
        super(myGPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)  # Tied weights
        self.padding_idx = padding_idx

        #self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.tau = tau

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    # def _prepare_encoder_decoder_kwargs_for_generation(
    #         self, inputs_tensor: torch.Tensor, model_kwargs, **kwargs
    # ):
    #     # check model_kwargs
    #     model_kwargs["encoder_hidden_states"]
    #

    # Reference: transformers/generation/utils.py
    # A function for beam-search decoding
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = True,        # Yoon-edited
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:

        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("token_type_ids") is not None:
            model_kwargs["token_type_ids"] = model_kwargs["token_type_ids"].repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat_interleave(expand_size, dim=0)

        # Yoon-edited
        is_encoder_decoder = True       # 일부러 내가 해놓은 것.
        if is_encoder_decoder:
            encoder_hidden_states = model_kwargs.get("encoder_hidden_states", None)
            if encoder_hidden_states is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_hidden_states` is defined.")
            expanded_encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                expand_size, dim=0
            )
            model_kwargs["encoder_hidden_states"] = expanded_encoder_hidden_states

            encoder_attention_mask = model_kwargs.get("encoder_attention_mask", None)
            if encoder_attention_mask is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_hidden_states` is defined.")
            expanded_encoder_attention_mask = encoder_attention_mask.repeat_interleave(
                expand_size, dim=0
            )
            model_kwargs["encoder_attention_mask"] = expanded_encoder_attention_mask

            decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
            if decoder_attention_mask is not None:
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask.repeat_interleave(expand_size, dim=0)

        return input_ids, model_kwargs


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **kwargs):
        # attention_mask: 여기서 attention_mask는 GPT2의 입력 단어에 관한 것이다. 지금까지 생성된 단어들에 대한 attn으로서 모두 1로 셋팅
        # cut decoder_input_ids if past is used
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values is not None:
            # only last token for inputs_ids if past is defined in kwargs
            input_ids = input_ids[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            #"pixel_values": kwargs.get("pixel_values", None),
            "encoder_hidden_states": kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": kwargs.get("encoder_attention_mask", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
        }

    # P-tuning 추가한 Forward 함수(230208)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,                   # [BOS] prefix
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,   # past
            attention_mask: Optional[torch.FloatTensor] = None,             # gpt2 input에 대한 attention_mask
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,              # Only activated for p-tuning
            encoder_hidden_states: Optional[torch.Tensor] = None,           # encoder_output
            encoder_attention_mask: Optional[torch.FloatTensor] = None,     # mask_encoder
            labels: Optional[torch.LongTensor] = None,                      # lm_labels
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            kwargs = None,
    ):

        # input_ids인 경우, inputs_embeds(p-tuning때) 경우
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            b_s, seq_len = input_ids.shape[:2]

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            b_s = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]

        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")



        #mask_queries = (input_ids != self.padding_idx).unsqueeze(-1).float()

        # 문제는 이새끼다.
        # train때는 정답 caption이 한번에 들어와서 attention_mask.size(1) == input_ids.size(1) 이어서 상관 없다.
        # inference: 항상 input_ids는 캡션 마지막 단어 하나만 들어오기 때문에 mask_queries는 항상 (batch, 1)가 되어야 한다.
        # 근데 이게 지금 attention_mask로 계산되서 문제다. attention_mask는 update 과정에서 그 길이가 늘어난다.
        # 그래서 음... 이걸 input_ids.size(1) == 1이면, input_ids로 mask_queries를 만들게끔 만들어 줘야 한다.
        # 대신 이게 될려면, training 과정에서 input_ids가 1인 경우는 없어야 한다... 없을 꺼야... 제발
        if seq_len == 1:
            # inference(generate 함수)
            mask_queries = torch.ones((b_s, seq_len), dtype=torch.float, device=attention_mask.device).unsqueeze(-1)
        else:
            # training / validation(input_ids == target_caption일 때 forward 호출 시)
            mask_queries = attention_mask.unsqueeze(-1).float()

        # mask_self_attention: 실제 GPT2Model Forward 함수에서 attention_mask를 만드는 과정: inference 일때도 attention_mask가 알아서 늘어나기 때문에 문제 없음
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input_ids.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        #mask_self_attention = mask_self_attention + (input_ids == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention + (attention_mask == 0).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        # if self._is_stateful:
        #     self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
        #     mask_self_attention = self.running_mask_self_attention

        # hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past,
        #                                            mask_queries=mask_queries, encoder_output=encoder_output,
        #                                            mask_encoder=mask_encoder, mask_self_attention=mask_self_attention,
        #                                            tau=self.tau)

        hidden_states, presents = self.transformer(input_ids=input_ids, position_ids=position_ids,
                                                   token_type_ids=token_type_ids, past_key_values=past_key_values,
                                                   mask_queries=mask_queries, encoder_output=encoder_hidden_states,
                                                   mask_encoder=encoder_attention_mask, mask_self_attention=mask_self_attention,
                                                   tau=self.tau)

        lm_logits = self.lm_head(hidden_states)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            return loss

        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return CausalLMOutputWithCrossAttentions(
            loss = None,
            logits = lm_logits,
            past_key_values = presents,
            hidden_states = hidden_states,
            attentions = None,
            cross_attentions = None,
        )



    # P-tuning 추가 전 forward 함수(~230207)
    def _forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,                   # [BOS] prefix
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,   # past
            attention_mask: Optional[torch.FloatTensor] = None,             # gpt2 input에 대한 attention_mask
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,              # Only activated for p-tuning
            encoder_hidden_states: Optional[torch.Tensor] = None,           # encoder_output
            encoder_attention_mask: Optional[torch.FloatTensor] = None,     # mask_encoder
            labels: Optional[torch.LongTensor] = None,                      # lm_labels
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            kwargs = None,
    ):
        b_s, seq_len = input_ids.shape[:2]
        #mask_queries = (input_ids != self.padding_idx).unsqueeze(-1).float()

        # 문제는 이새끼다.
        # train때는 정답 caption이 한번에 들어와서 attention_mask.size(1) == input_ids.size(1) 이어서 상관 없다.
        # inference: 항상 input_ids는 캡션 마지막 단어 하나만 들어오기 때문에 mask_queries는 항상 (batch, 1)가 되어야 한다.
        # 근데 이게 지금 attention_mask로 계산되서 문제다. attention_mask는 update 과정에서 그 길이가 늘어난다.
        # 그래서 음... 이걸 input_ids.size(1) == 1이면, input_ids로 mask_queries를 만들게끔 만들어 줘야 한다.
        # 대신 이게 될려면, training 과정에서 input_ids가 1인 경우는 없어야 한다... 없을 꺼야... 제발
        if seq_len == 1:
            # inference(generate 함수)
            mask_queries = torch.ones((b_s, seq_len), dtype=torch.float, device=attention_mask.device).unsqueeze(-1)
        else:
            # training / validation(input_ids == target_caption일 때 forward 호출 시)
            mask_queries = attention_mask.unsqueeze(-1).float()

        # mask_self_attention: 실제 GPT2Model Forward 함수에서 attention_mask를 만드는 과정: inference 일때도 attention_mask가 알아서 늘어나기 때문에 문제 없음
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input_ids.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        #mask_self_attention = mask_self_attention + (input_ids == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention + (attention_mask == 0).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        # if self._is_stateful:
        #     self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
        #     mask_self_attention = self.running_mask_self_attention

        # hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past,
        #                                            mask_queries=mask_queries, encoder_output=encoder_output,
        #                                            mask_encoder=mask_encoder, mask_self_attention=mask_self_attention,
        #                                            tau=self.tau)

        hidden_states, presents = self.transformer(input_ids=input_ids, position_ids=position_ids,
                                                   token_type_ids=token_type_ids, past_key_values=past_key_values,
                                                   mask_queries=mask_queries, encoder_output=encoder_hidden_states,
                                                   mask_encoder=encoder_attention_mask, mask_self_attention=mask_self_attention,
                                                   tau=self.tau)

        lm_logits = self.lm_head(hidden_states)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            return loss

        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return CausalLMOutputWithCrossAttentions(
            loss = None,
            logits = lm_logits,
            past_key_values = presents,
            hidden_states = hidden_states,
            attentions = None,
            cross_attentions = None,
        )

        #return lm_logits, presents

class GPT2LMHead(Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2Model(Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        #self.register_state('running_seq', torch.zeros((1,)).long())

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def get_input_embeddings(self):
        return self.wte

    def forward(self, input_ids=None, inputs_embeds= None, position_ids=None, token_type_ids=None, past_key_values=None,
                mask_queries=None, encoder_output=None, mask_encoder=None, mask_self_attention=None, tau=0):

        # input_ids인 경우, inputs_embeds(p-tuning때) 경우
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            b_s, seq_len = input_ids.shape[:2]

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            b_s = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]

        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            # position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
            #                             device=input_ids.device)
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long,
                                        device=device)
            #position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_ids.size(-1))
            position_ids = position_ids.view(-1, position_ids.size(-1))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds     # broadcastable
        presents = []

        for block, layer_past in zip(self.h, past_key_values):
            hidden_states, present = block(hidden_states, layer_past, mask_queries=mask_queries,
                                           encoder_output=encoder_output, mask_encoder=mask_encoder,
                                           mask_self_attention=mask_self_attention, tau=tau)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        test = x.contiguous().view(-1, x.size(-1))
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight)

        x = x.view(*size_out)
        return x


class Attention(Module):
    def __init__(self, nx, n_ctx, config, scale=False, can_be_stateful=False):
        super(Attention, self).__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd) ; GPT2-Medium: 1024
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        #self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.can_be_stateful = can_be_stateful
        self.attn_pdrop = nn.Dropout(config.attn_pdrop)
        #
        # if self.can_be_stateful:
        #     self.register_state('running_keys', torch.zeros((12, 0, 64)))
        #     self.register_state('running_values', torch.zeros((12, 0, 64)))

    def _attn(self, q, k, v, mask_self_attention):

        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))

        if mask_self_attention is not None:
            w = w.masked_fill(mask_self_attention, -10000.0)
            # w[:,:,:,:nk] = w[:,:,:,:nk].masked_fill(mask_self_attention, -1e7)
        # nd, ns = w.size(-2), w.size(-1)
        # b = self.bias[:, :, ns-nd:ns, :ns]

        # w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        self.w = self.attn_pdrop(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, mask_self_attention=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        # if self.can_be_stateful and self._is_stateful:
        #     self.running_keys = torch.cat([self.running_keys, key.transpose(-2, -1)], -2)
        #     key = self.running_keys.transpose(-2, -1)
        #
        #     self.running_values = torch.cat([self.running_values, value], -2)
        #     value = self.running_values
        # GPT2Attention의 Forward 함수에서 처리했었음.    inference시 input_ids는 항상 1개이기 때문에 past_key_value에 이전 입력에 대한 key/value값이 저장되어 있음
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, mask_self_attention)
        a = self.merge_heads(a)
        a = self.c_proj(a)

        return a, present


# GPT2-Medium
class Enc_Dec_Attention(Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Enc_Dec_Attention, self).__init__()
        #n_state = nx = 768
        n_state = nx        # nx: n_embd; GPT2: 768, GPT2-Medium: 1024
        #n_ctx = 60         # Yoon-edited : original GPT2로
        scale = True
        #n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]

        #assert n_state % 12 == 0
        assert n_state % config.n_head == 0

        #self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        #self.n_head = 12
        self.n_head = config.n_head     # gpt2: 12; gpt2-medium: 16
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

        #self.fc_q = nn.Linear(n_state, 64 * 12)
        #self.fc_k = nn.Linear(n_state, 64 * 12)
        #self.fc_v = nn.Linear(n_state, 64 * 12)
        self.fc_q = nn.Linear(n_state, n_state)     # Yoon-edited
        self.fc_k = nn.Linear(n_state, n_state)     # Yoon-edited
        self.fc_v = nn.Linear(n_state, n_state)     # Yoon-edited

        self.attn_dropout = nn.Dropout(0.2)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        # nn.init.xavier_uniform_(self.fc_o.weight)

    def _attn(self, q, k, v, enc_dec_attention):
        nk = k.shape[-1]
        #print(q.size())
        #print(k.size())
        w = torch.matmul(q, k)      # q: batch, nh, seq_len(18), feat/nh; k: batch, nh, feat/nh, #frames
        if self.scale:              # w: batch, nh, seq_len, #frames(12)
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        #b = self.bias[:, :, ns - nd:ns, :ns]
        if enc_dec_attention is not None:
            w = w.masked_fill(enc_dec_attention, -10000.0)
            # w[:, :, ns-nd:ns, :ns] = w[:, :, ns-nd:ns, :ns].masked_fill(enc_dec_attention, -1e10)

        # w = w*enc_dec_attention

        # w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, encoder_output=None, mask_encoder=None):
        # encoder_output: batch, #frames, feat
        query = self.fc_q(x)            # x == query: batch, seq_len, feat
        encoder_key = self.fc_k(encoder_output)     # encoder_key: batch, #frames, feat
        encoder_value = self.fc_v(encoder_output)   # encoder_value: batch, #frames, feat
        query = self.split_heads(query)             # query: batch, nh, seq_len, feat/nh
        encoder_key = self.split_heads(encoder_key, k=True) # encoder_key: batch, nh, feat/nh, #frames
        encoder_value = self.split_heads(encoder_value)     # encoder_value: batch, nh, #frames, feat/nh

        a = self._attn(query, encoder_key, encoder_value, mask_encoder)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a



# GPT2-small
class _Enc_Dec_Attention(Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Enc_Dec_Attention, self).__init__()
        n_state = nx = 768
        n_ctx = 60
        scale = True
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % 12 == 0
        #self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = 12
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

        self.fc_q = nn.Linear(n_state, 64 * 12)
        self.fc_k = nn.Linear(n_state, 64 * 12)
        self.fc_v = nn.Linear(n_state, 64 * 12)

        self.attn_dropout = nn.Dropout(0.2)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        # nn.init.xavier_uniform_(self.fc_o.weight)

    def _attn(self, q, k, v, enc_dec_attention):
        nk = k.shape[-1]
        #print(q.size())
        #print(k.size())
        w = torch.matmul(q, k)      # q: batch, nh, seq_len(18), feat/nh; k: batch, nh, feat/nh, #frames
        if self.scale:              # w: batch, nh, seq_len, #frames(12)
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        #b = self.bias[:, :, ns - nd:ns, :ns]
        if enc_dec_attention is not None:
            w = w.masked_fill(enc_dec_attention, -10000.0)
            # w[:, :, ns-nd:ns, :ns] = w[:, :, ns-nd:ns, :ns].masked_fill(enc_dec_attention, -1e10)

        # w = w*enc_dec_attention

        # w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, encoder_output=None, mask_encoder=None):
        # encoder_output: batch, #frames, feat
        query = self.fc_q(x)            # x == query: batch, seq_len, feat
        encoder_key = self.fc_k(encoder_output)     # encoder_key: batch, #frames, feat
        encoder_value = self.fc_v(encoder_output)   # encoder_value: batch, #frames, feat
        query = self.split_heads(query)             # query: batch, nh, seq_len, feat/nh
        encoder_key = self.split_heads(encoder_key, k=True) # encoder_key: batch, nh, feat/nh, #frames
        encoder_value = self.split_heads(encoder_value)     # encoder_value: batch, nh, #frames, feat/nh

        a = self._attn(query, encoder_key, encoder_value, mask_encoder)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd

        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale, can_be_stateful=True)
        self.enc_dec_attn = Enc_Dec_Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.resid_pdrop = nn.Dropout(config.resid_pdrop)

        self.fc_alpha1 = nn.Linear(nx + nx, nx)
        self.fc_alpha2 = nn.Linear(nx + nx, nx)
        self.fc_alpha3 = nn.Linear(nx + nx, nx)

    def forward(self, x, layer_past=None, mask_queries=None, encoder_output=None, mask_encoder=None,
                mask_self_attention=None, tau=0):
        threshold = tau
        # Step1. masked self attention for generated captions for each time-stamp
        self_attention, present = self.attn(self.ln_1(x), layer_past=layer_past,
                                            mask_self_attention=mask_self_attention)
        a = x + self_attention
        a = self.resid_pdrop(a)
        # Step2. CrossAttention Layer whose query comes from step1, and key and value come from the encoder
        enc_att1 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 0]),
                                     mask_encoder=mask_encoder)

        enc_att2 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 1]),
                                     mask_encoder=mask_encoder)

        enc_att3 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 2]),
                                     mask_encoder=mask_encoder)

        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([a, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([a, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([a, enc_att3], -1)))

        linguistics_alpha1_mask = torch.where(alpha1 > threshold, torch.ones_like(alpha1), torch.zeros_like(alpha1))
        linguistics_alpha2_mask = torch.where(alpha2 > threshold, torch.ones_like(alpha2), torch.zeros_like(alpha2))
        linguistics_alpha3_mask = torch.where(alpha3 > threshold, torch.ones_like(alpha3), torch.zeros_like(alpha3))

        visual_alpha1_mask = torch.where(alpha1 < 1 - threshold, torch.ones_like(alpha1), torch.zeros_like(alpha1))
        visual_alpha2_mask = torch.where(alpha2 < 1 - threshold, torch.ones_like(alpha2), torch.zeros_like(alpha2))
        visual_alpha3_mask = torch.where(alpha3 < 1 - threshold, torch.ones_like(alpha3), torch.zeros_like(alpha3))

        enc_att1 = alpha1 * linguistics_alpha1_mask * a + (1 - alpha1) * visual_alpha1_mask * enc_att1
        enc_att2 = alpha2 * linguistics_alpha2_mask * a + (1 - alpha2) * visual_alpha2_mask * enc_att2
        enc_att3 = alpha3 * linguistics_alpha3_mask * a + (1 - alpha3) * visual_alpha3_mask * enc_att3

        enc_att = (enc_att1 + enc_att2 + enc_att3) / np.sqrt(3)
        a = enc_att * mask_queries  # PAD는 masking 처리

        m = self.mlp(self.ln_2(a))

        encoder_result = a + m      # residual connection

        encoder_result = self.resid_pdrop(encoder_result)

        encoder_result = encoder_result * mask_queries      # residual 이니까 다시 masking queries


        return encoder_result, present
