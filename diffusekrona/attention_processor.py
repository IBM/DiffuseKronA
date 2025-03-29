# Note: This file is a modified version of the original file from the diffusers library.

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from krona import KronALinearLayer

from diffusers.models.attention_processor import *
""" Current krona is the clice LoRA version. It's not official KronA codebase """
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class LoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None, 
            adapter_type = None,
            attn_update_unet = None,
            **kwargs,
        ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank # default is k rank

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        self.attn_update_unet = list(attn_update_unet)
        
        if(adapter_type == "lora"):
            if("q" in attn_update_unet):
                self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
            if("k" in attn_update_unet):
                self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
            if("v" in attn_update_unet):
                self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
            if("o" in attn_update_unet):
                self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

        elif(adapter_type == "krona"):
            raise ValueError("Currently not supported.")
        
        else:
            raise ValueError("Only lora and krona supported.")

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if "q" in self.attn_update_unet: query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        else: query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if "k" in self.attn_update_unet: key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        else: key = attn.to_k(encoder_hidden_states)
       
        if "v" in self.attn_update_unet: value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        else: value = attn.to_v(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        if "o" in self.attn_update_unet: hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        else: hidden_states = attn.to_out[0](hidden_states)
        
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



class LoRAAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism using PyTorch 2.0's memory-efficient scaled dot-product
    attention.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, network_alpha=None, 
            adapter_type = None,
            attn_update_unet = None,
            k_rank=None,
            q_rank=None,
            v_rank=None,
            out_rank=None,
            **kwargs,
        ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        self.attn_update_unet = list(attn_update_unet)
        
        if(adapter_type == "lora"):
            if("q" in attn_update_unet):
                self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
            if("k" in attn_update_unet):
                self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, k_rank, network_alpha)
            if("v" in attn_update_unet):
                self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
            if("o" in attn_update_unet):
                self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)
            
        elif(adapter_type == "krona"):
            if("q" in attn_update_unet):
                self.to_q_lora = KronALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
            if("k" in attn_update_unet):
                self.to_k_lora = KronALinearLayer(cross_attention_dim or hidden_size, hidden_size, k_rank, network_alpha)
            if("v" in attn_update_unet):
                self.to_v_lora = KronALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
            if("o" in attn_update_unet):
                self.to_out_lora = KronALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)
            
        else:
            raise ValueError("Only lora & krona supported.")
    
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if("q" in self.attn_update_unet): query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        else: query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if("k" in self.attn_update_unet): key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        else: key = attn.to_k(encoder_hidden_states)

        if("v" in self.attn_update_unet): value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        else: value = attn.to_v(encoder_hidden_states)
        
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        if("o" in self.attn_update_unet): hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        else: hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


AttentionProcessor = Union[
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
]

LORA_ATTENTION_PROCESSORS = (
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
)
