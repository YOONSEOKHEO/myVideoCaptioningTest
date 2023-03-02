#-*- coding: utf-8 -*-
# Implemented by Yoonseok Heo at 230112
# Email: nlp.ysheo419@gmail.com
# Code Reference URL: https://github.com/ju-chen/Efficient-Prompt

import torch, einops, math
import torch.nn as nn
from collections import OrderedDict
from transformers.configuration_utils import PretrainedConfig
from transformers import ViTConfig
from transformers.utils import ModelOutput
from typing import Optional, Tuple

class MyVideoEncoderConfig(ViTConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = kwargs.pop("decoder_hidden_size", None)
        if self.hidden_size is None:
            raise ValueError("No decoder_hidden_size in your video config file")
        self.decoder_hidden_size = self.hidden_size
        self.max_frames = kwargs.pop("max_frames", 25)
        self.tfm_heads = kwargs.pop("tfm_heads", 8)
        self.tfm_layers = kwargs.pop("tfm_layers", 3)
        self.video_dropout = kwargs.pop("video_dropout", 0.1)
        self.d_in = kwargs.pop("d_in", 512)
        self.cuda_device = "cuda" if torch.cuda.is_available() else "cpu"


class MyVideoModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor] = None
    attention_mask: torch.FloatTensor = None


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()

        # batch_first: False: (seq_len, batch, feat_dim) 값이 forward로 들어와야함
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=False)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)


    def make_pad_mask(self, query, key, pad_value = True):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.eq(pad_value).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)  # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.eq(pad_value).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask

    def attention(self, x: torch.Tensor, attention_mask: torch.Tensor):
        """
        :param x: frame embeddings (padded_seq_len, batch_size, feat_dim)
        :param attention_mask: (batch_size, 1, 1, padded_seq_len)
        :return:
        """

        # attn_mask : (batch_size * num_heads), query_seq_len, key_seq_len
        # Video Encoder에서는 query_seq_len == key_seq_len == padded_seq_len

        if attention_mask is not None:
            attn_mask = self.make_pad_mask(query=attention_mask, key=attention_mask)       # attn_mask: batch, 1, seq_len, seq_len
            attn_mask = attn_mask.repeat(1, self.attn.num_heads, 1, 1)      # attn_mask: batch, num_heads, seq_len, seq_len
            attn_mask = attn_mask.reshape(-1, attn_mask.size(-2), attn_mask.size(-1))   # attn_mask: batch*num_heads, seq_len, seq_len

            attn_mask = attn_mask.to(dtype=bool, device=x.device)
        else:
            attn_mask = None

        #self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        #attn_mask = attention_mask.to(dtype=x.dtype, device=x.device) if attention_mask is not None else None

        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0], attn_mask

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):     # x: max_frames(16), batch, clip_image_feat(512)
        attention_output, attn_mask = self.attention(self.ln_1(x), attention_mask)
        x = x + attention_output
        x = x + self.mlp(self.ln_2(x))
        return x, attention_mask


class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        # self.resblocks = nn.Sequential(
        #     #*[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])
        #     *[ResidualAttentionBlock(width, heads, dropout) for _ in range(layers)])
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, dropout) for _ in range(layers)]
        )


    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        #return self.resblocks(x, attention_mask)
        layer_wise_encoder_outputs_list = []
        for idx, block in enumerate(self.resblocks):
            # x: #frames(25), batch, feat_dim(512)
            x, attention_mask = block(x = x, attention_mask=attention_mask)      # attn_mask = (batch, 1, #frames, #frames)
            layer_wise_encoder_outputs_list.append(x)

        layer_wise_encoder_outputs = torch.stack(layer_wise_encoder_outputs_list, dim=0).permute((2, 0, 1, 3))

        # layer_wise_encoder_outputs: (batch, args.tfm_layers, #max_frames, feats)
        return layer_wise_encoder_outputs, attention_mask.unsqueeze(1).unsqueeze(1)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MyVideoEncoder(torch.nn.Module):
    def __init__(self, config: MyVideoEncoderConfig):
        super(MyVideoEncoder, self).__init__()
        self.main_input_name = "pixel_values"
        self.config = config
        self.max_frames = config.max_frames
        self.cuda_device = config.cuda_device
        self.has_feature_input = True   # 모든 비디오에 대한 FRAME-Level CLIP Embedding 사전 추출한 것을 사용하는 경우 TRUE
        self.hidden_size = config.decoder_hidden_size
        self.tfm_layers = config.tfm_layers
        self.tfm_heads = config.tfm_heads
        self.dropout = config.video_dropout
        self.d_in = config.d_in         # CLIP output feature

        # Feature Transform to hidden_size
        self.fc = nn.Linear(self.d_in, self.hidden_size)
        self.fc_dropout = nn.Dropout(p=self.dropout)
        self.fc_ln = nn.LayerNorm(self.hidden_size)

        # Temporal Modelling
        self.temporalEmbedding = torch.nn.Embedding(self.max_frames, self.hidden_size)
        self.temporalModelling = TemporalModelling(width=self.hidden_size, layers=self.tfm_layers,
                                                   heads=self.tfm_heads, dropout=self.dropout)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)
        for p in self.fc.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_output_embeddings(self):
        return None

    def forward(self, pixel_values, attention_mask = None, **kwargs):
        """
        :param pixel_values:
        :param attention_mask: attention_mask over the video frames
        :return: vFeature - a tuple(layer-wise encoder outputs: batch, #layers, #max_frames, feat_dim,
                                    attention_mask of frames: batch, 1, 1, #max_frames ==> Corresponding to decoder)
        """
        num_frames = pixel_values.size(1)
        # transform feature dimension into GPT2
        out = gelu(self.fc(pixel_values.float()))       # float16 ==> float32
        out = self.fc_dropout(out)
        transformed_vids = self.fc_ln(out)     # batch, #max_frames, 768(1024)

        # encode videos
        if self.has_feature_input:
            vFeature = einops.rearrange(transformed_vids.float(), 'b t c -> t b c', t=num_frames)
        """
        else:
            iFeature = self.clipmodel.encode_image(einops.rearrange(vids, 'b t c h w -> (b t) c h w'))
            vFeature = einops.rearrange(iFeature, '(b t) c -> t b c', t=num_frames)
        """

        tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(num_frames).to(self.cuda_device)),
                                      't c -> t b c', b=vFeature.size(1))
        vFeature = vFeature + tempEmbedding.to(self.cuda_device)
        vFeature = self.temporalModelling(vFeature, attention_mask)
        #vFeature = vFeature.mean(dim=0)

        return MyVideoModelOutput(
            last_hidden_state   = vFeature[0][-1],
            hidden_states       = vFeature[0],
            attention_mask      = vFeature[1]
        )
        #return vFeature


class VideoEncoder(torch.nn.Module):
    def __init__(self, args, decoder_hidden_size, max_frames, device, d_in=512):
        super(VideoEncoder, self).__init__()
        self.args = args
        self.max_frames = max_frames
        self.cuda_device = device
        self.has_feature_input = True   # 모든 비디오에 대한 FRAME-Level CLIP Embedding 사전 추출한 것을 사용하는 경우 TRUE
        self.hidden_size = decoder_hidden_size
        self.tfm_layers = args.tfm_layers
        self.tfm_heads = args.tfm_heads
        self.dropout = 0.1
        self.d_in = d_in        # CLIP output feature

        # Feature Transform to hidden_size
        self.fc = nn.Linear(d_in, self.hidden_size)
        self.fc_dropout= nn.Dropout(p=self.dropout)
        self.fc_ln = nn.LayerNorm(self.hidden_size)

        # Temporal Modelling
        self.temporalEmbedding = torch.nn.Embedding(self.max_frames, self.hidden_size)
        self.temporalModelling = TemporalModelling(width=self.hidden_size, layers=self.tfm_layers,
                                                   heads=self.tfm_heads, dropout=self.dropout)

        self.initialize_parameters()


    def initialize_parameters(self):
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)

    def forward(self, pixel_values, encoder_video_attention_mask = None, **kwargs):
        """
        :param pixel_values:
        :param encoder_video_attention_mask:
        :return: vFeature - a tuple(layer-wise encoder outputs: batch, #layers, #max_frames, feat_dim,
                                    attention_mask of frames: batch, 1, 1, #max_frames ==> Corresponding to decoder)
        """
        num_frames = pixel_values.size(1)
        # transform feature dimension into GPT2
        out = gelu(self.fc(pixel_values.float()))       # float16 ==> float32
        out = self.fc_dropout(out)
        transformed_vids = self.fc_ln(out)     # batch, #max_frames, 768(1024)

        # encode videos
        if self.has_feature_input:
            vFeature = einops.rearrange(transformed_vids.float(), 'b t c -> t b c', t=num_frames)
        """
        else:
            iFeature = self.clipmodel.encode_image(einops.rearrange(vids, 'b t c h w -> (b t) c h w'))
            vFeature = einops.rearrange(iFeature, '(b t) c -> t b c', t=num_frames)
        """

        tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(num_frames).to(self.cuda_device)),
                                      't c -> t b c', b=vFeature.size(1))
        vFeature = vFeature + tempEmbedding.to(self.cuda_device)
        vFeature = self.temporalModelling(vFeature, encoder_video_attention_mask)
        #vFeature = vFeature.mean(dim=0)

        return vFeature



