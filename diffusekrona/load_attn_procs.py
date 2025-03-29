# Note: This file is a modified version of the original file from the diffusers library.

import diffusers
from typing import Callable, Dict, List, Optional, Union
import torch
import safetensors
import warnings
import copy, os
import torch.nn.functional as F
from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    _get_model_file,
    logging,
)

from collections import defaultdict
logger = logging.get_logger(__name__)
LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"


def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
    r"""
    Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
    defined in
    [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
    and be a `torch.nn.Module` class.

    Parameters:
        pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
            Can be either:

                - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                    the Hub.
                - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                    with [`ModelMixin.save_pretrained`].
                - A [torch state
                    dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
            incompletely downloaded files are deleted.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        subfolder (`str`, *optional*, defaults to `""`):
            The subfolder location of a model file within a larger model repository on the Hub or locally.
        mirror (`str`, *optional*):
            Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
            guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
            information.

    """
    from diffusers.models.attention_processor import (
        AttnAddedKVProcessor,
        AttnAddedKVProcessor2_0,
        CustomDiffusionAttnProcessor,
        LoRAAttnAddedKVProcessor,
        LoRAXFormersAttnProcessor,
        SlicedAttnAddedKVProcessor,
        XFormersAttnProcessor,
    )
    from attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0

    from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear, LoRAConv2dLayer, LoRALinearLayer
    from krona import KronALinearLayer
    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)
    # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
    # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    network_alphas = kwargs.pop("network_alphas", None)
    is_network_alphas_none = network_alphas is None
    attn_update_unet = list(os.getenv("attn_update_unet"))
    adapter_type = os.getenv("adapter_type")
    allow_pickle = False

    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        # Let's first try to load .safetensors weights
        if (use_safetensors and weight_name is None) or (
            weight_name is not None and weight_name.endswith(".safetensors")
        ):
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = safetensors.torch.load_file(model_file, device="cpu")
            except IOError as e:
                if not allow_pickle:
                    raise e
                # try loading non-safetensors weights
                pass
        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or LORA_WEIGHT_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = torch.load(model_file, map_location="cpu")
    else:
        state_dict = pretrained_model_name_or_path_or_dict

    # fill attn processors
    attn_processors = {}
    non_attn_lora_layers = []

    is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys())
    is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())

    if is_lora:
        is_new_lora_format = all(
            key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in state_dict.keys()
        )
        if is_new_lora_format:
            # Strip the `"unet"` prefix.
            is_text_encoder_present = any(key.startswith(self.text_encoder_name) for key in state_dict.keys())
            if is_text_encoder_present:
                warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
                warnings.warn(warn_message)
            unet_keys = [k for k in state_dict.keys() if k.startswith(self.unet_name)]
            state_dict = {k.replace(f"{self.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

        lora_grouped_dict = defaultdict(dict)
        mapped_network_alphas = {}

        all_keys = list(state_dict.keys())
        
        for key in all_keys:
            value = state_dict.pop(key)
            attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
            lora_grouped_dict[attn_processor_key][sub_key] = value

            # Create another `mapped_network_alphas` dictionary so that we can properly map them.
            if network_alphas is not None:
                network_alphas_ = copy.deepcopy(network_alphas)
                for k in network_alphas_:
                    if k.replace(".alpha", "") in key:
                        mapped_network_alphas.update({attn_processor_key: network_alphas.pop(k)})

        if not is_network_alphas_none:
            if len(network_alphas) > 0:
                raise ValueError(
                    f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                )

        if len(state_dict) > 0:
            raise ValueError(
                f"The `state_dict` has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
            )
        
        for key, value_dict in lora_grouped_dict.items():
            attn_processor = self
            for sub_key in key.split("."):
                attn_processor = getattr(attn_processor, sub_key)

            # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
            # or add_{k,v,q,out_proj}_proj_lora layers.
            if "lora.down.weight" in value_dict:
                if(adapter_type=="lora"): rank = value_dict["lora.down.weight"].shape[0]
                elif(adapter_type=="krona"): raise ValueError("Currently not supported.")
                else: raise ValueError("Only LoRA and KronA supported.")

                if isinstance(attn_processor, LoRACompatibleConv):
                    in_features = attn_processor.in_channels
                    out_features = attn_processor.out_channels
                    kernel_size = attn_processor.kernel_size

                    lora = LoRAConv2dLayer(
                        in_features=in_features,
                        out_features=out_features,
                        rank=rank,
                        kernel_size=kernel_size,
                        stride=attn_processor.stride,
                        padding=attn_processor.padding,
                        network_alpha=mapped_network_alphas.get(key),
                    )
                elif isinstance(attn_processor, LoRACompatibleLinear):
                    lora = LoRALinearLayer(
                        attn_processor.in_features,
                        attn_processor.out_features,
                        rank,
                        mapped_network_alphas.get(key),
                    )
                else:
                    raise ValueError(f"Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module.")

                value_dict = {k.replace("lora.", ""): v for k, v in value_dict.items()}
                lora.load_state_dict(value_dict)
                non_attn_lora_layers.append((attn_processor, lora))

            elif "lora_layer.down.weight" in value_dict:
                if(adapter_type=="lora"): rank = value_dict["lora_layer.down.weight"].shape[0]
                elif(adapter_type=="krona"):
                    rank_a2, rank_a1 = value_dict["lora_layer.down.weight"].shape # A
                    rank_b1, rank_b2 = value_dict[f"lora_layer.up.weight"].shape # B
                    rank = (rank_a1, rank_a2)
                    hidden_size = rank_a1 * rank_b1 # in_features
                        
                else: raise ValueError("Only LoRA and KronA supported.")

                if isinstance(attn_processor, LoRACompatibleConv):
                    in_features = attn_processor.in_channels
                    out_features = attn_processor.out_channels
                    kernel_size = attn_processor.kernel_size

                    lora = LoRAConv2dLayer(
                        in_features=in_features,
                        out_features=out_features,
                        rank=rank,
                        kernel_size=kernel_size,
                        stride=attn_processor.stride,
                        padding=attn_processor.padding,
                        network_alpha=mapped_network_alphas.get(key),
                    )
                    # ToDo: need to add krona here as well in future.
                elif isinstance(attn_processor, LoRACompatibleLinear):
                    if adapter_type =="lora":
                        lora = LoRALinearLayer(
                            attn_processor.in_features,
                            attn_processor.out_features,
                            rank,
                            mapped_network_alphas.get(key),
                            device=self.device, 
                            dtype=self.dtype,
                        )
                    elif adapter_type == "krona":
                        lora = KronALinearLayer(
                            attn_processor.in_features,
                            attn_processor.out_features,
                            rank,
                            mapped_network_alphas.get(key),
                            device=self.device, 
                            dtype=self.dtype,
                        )
                    else:
                        raise AttributeError(f"Currently only LoRA and KronA are supported.")
                else:
                    raise ValueError(f"Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module.")

                value_dict = {k.replace("lora_layer.", ""): v for k, v in value_dict.items()}
                lora.load_state_dict(value_dict)
                non_attn_lora_layers.append((attn_processor, lora))
            
            else: 
                # To handle SDXL.
                
                rank_mapping = {}
                hidden_size_mapping = {}
                projection_ids_list = []
                if("k" in attn_update_unet): projection_ids_list.append("to_k")
                if("q" in attn_update_unet): projection_ids_list.append("to_q")
                if("v" in attn_update_unet): projection_ids_list.append("to_v")
                if("o" in attn_update_unet): projection_ids_list.append("to_out")

                for projection_id in projection_ids_list:

                    # Added lora and KronA
                    if(adapter_type=="lora"):
                        rank = value_dict[f"{projection_id}_lora.down.weight"].shape[0]
                        hidden_size = value_dict[f"{projection_id}_lora.up.weight"].shape[0]
                    elif(adapter_type=="krona"):
                        rank_a2, rank_a1 = value_dict[f"{projection_id}_lora.down.weight"].shape # A
                        rank_b1, rank_b2 = value_dict[f"{projection_id}_lora.up.weight"].shape # B
                        rank = (rank_a1, rank_a2)
                        hidden_size = rank_a1 * rank_b1 # in_features
                    else: raise ValueError("Only LoRA and KronA are supported.")

                    rank_mapping.update({f"{projection_id}_lora.down.weight": rank})
                    hidden_size_mapping.update({f"{projection_id}_lora.up.weight": hidden_size})
                    

                """ For cross attention dimention """
                if isinstance(
                    attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)
                ):
                    if("k" in attn_update_unet): _, dim_ = value_dict["add_k_proj_lora.down.weight"].size()
                    elif("v" in attn_update_unet): _, dim_ = value_dict["add_v_proj_lora.down.weight"].size()
                    elif("q" in attn_update_unet): _, dim_ = value_dict["add_q_proj_lora.down.weight"].size()
                    elif("o" in attn_update_unet): _, dim_ = value_dict["add_out_proj_lora.down.weight"].size()
                    else: raise ValueError("attention weight type error.")
                    
                    # Added lora and KronA
                    if(adapter_type=='lora'): cross_attention_dim = dim_
                    elif(adapter_type=='krona'): raise ValueError("Currently not supported.")
                    else: raise ValueError("Only LoRA and KronA are supported.")
                    attn_processor_class = LoRAAttnAddedKVProcessor
                else:
                    # Added lora and KronA
                    if(adapter_type=='lora'): 
                        if("k" in attn_update_unet): _, dim_ = value_dict["to_k_lora.down.weight"].size()
                        elif("v" in attn_update_unet): _, dim_ = value_dict["to_v_lora.down.weight"].size()
                        elif("q" in attn_update_unet): _, dim_ = value_dict["to_q_lora.down.weight"].size()
                        elif("o" in attn_update_unet): _, dim_ = value_dict["to_out_lora.down.weight"].size()
                        else: raise ValueError("attention weight type error.")
                        cross_attention_dim = dim_
                    
                    elif(adapter_type=='krona'): 
                        if("k" in attn_update_unet): 
                            rank_a2, rank_a1 = value_dict["to_k_lora.down.weight"].size() # A
                            rank_b1, rank_b2 = value_dict["to_k_lora.up.weight"].size() # B
                            cross_attention_dim = rank_a2 * rank_b2
                        elif("v" in attn_update_unet):
                            rank_a2, rank_a1 = value_dict["to_v_lora.down.weight"].size() # A
                            rank_b1, rank_b2 = value_dict["to_v_lora.up.weight"].size() # B
                            cross_attention_dim = rank_a2 * rank_b2
                        elif("q" in attn_update_unet): 
                            rank_a2, rank_a1 = value_dict["to_q_lora.down.weight"].size() # A
                            rank_b1, rank_b2 = value_dict["to_q_lora.up.weight"].size() # B
                            cross_attention_dim = rank_a2 * rank_b2
                        elif("o" in attn_update_unet): 
                            rank_a2, rank_a1 = value_dict["to_out_lora.down.weight"].size() # A 
                            rank_b1, rank_b2 = value_dict["to_out_lora.up.weight"].size() # A 
                            cross_attention_dim = rank_a2 * rank_b2
                        else: raise ValueError("attention weight type error.")
                        
                        # raise ValueError("Currently not supported.")
                    else: raise ValueError("Only LoRA and KronA are supported.")
                    
                    if isinstance(attn_processor, (XFormersAttnProcessor, LoRAXFormersAttnProcessor)):
                        attn_processor_class = LoRAXFormersAttnProcessor
                    else:
                        attn_processor_class = (
                            LoRAAttnProcessor2_0
                            if hasattr(F, "scaled_dot_product_attention")
                            else LoRAAttnProcessor
                        )
                
                if("k" in attn_update_unet): 
                    k_rank = rank_mapping.get("to_k_lora.down.weight")
                    hidden_size_ = hidden_size_mapping.get("to_k_lora.up.weight")
                if("q" in attn_update_unet): 
                    q_rank = rank_mapping.get("to_q_lora.down.weight")
                    hidden_size_ = hidden_size_mapping.get("to_q_lora.up.weight")
                if("v" in attn_update_unet): 
                    v_rank = rank_mapping.get("to_v_lora.down.weight")
                    hidden_size_ = hidden_size_mapping.get("to_v_lora.up.weight")
                if("o" in attn_update_unet): 
                    out_rank = rank_mapping.get("to_out_lora.down.weight")
                    hidden_size_ = hidden_size_mapping.get("to_out_lora.up.weight")
                
                
                if attn_processor_class is not LoRAAttnAddedKVProcessor: # getting call
                    attn_processors[key] = attn_processor_class(
                        k_rank=k_rank if "k" in attn_update_unet else None, # added
                        q_rank=q_rank if "q" in attn_update_unet else None, # added
                        v_rank=v_rank if "v" in attn_update_unet else None,  # added
                        out_rank=out_rank if "o" in attn_update_unet else None, # added
                        hidden_size=hidden_size_,
                        cross_attention_dim=cross_attention_dim,
                        network_alpha=mapped_network_alphas.get(key),
                        q_hidden_size=hidden_size_mapping.get("to_q_lora.up.weight"),
                        v_hidden_size=hidden_size_mapping.get("to_v_lora.up.weight"),
                        out_hidden_size=hidden_size_mapping.get("to_out_lora.up.weight"),
                        adapter_type=adapter_type,
                        attn_update_unet=attn_update_unet,
                    )
                else:
                    attn_processors[key] = attn_processor_class(
                        k_rank=k_rank, # added
                        q_rank=q_rank, # added
                        v_rank=v_rank, # added
                        out_rank=out_rank, # added
                        hidden_size=hidden_size_,
                        cross_attention_dim=cross_attention_dim,
                        network_alpha=mapped_network_alphas.get(key),
                        adapter_type=adapter_type,
                        attn_update_unet=attn_update_unet,
                    )
                
                attn_processors[key].load_state_dict(value_dict)
                    

    elif is_custom_diffusion:
        custom_diffusion_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            if len(value) == 0:
                custom_diffusion_grouped_dict[key] = {}
            else:
                if "to_out" in key:
                    attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                else:
                    attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

        for key, value_dict in custom_diffusion_grouped_dict.items():
            if len(value_dict) == 0:
                attn_processors[key] = CustomDiffusionAttnProcessor(
                    train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                )
            else:
                cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
                hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
                train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                attn_processors[key] = CustomDiffusionAttnProcessor(
                    train_kv=True,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
                attn_processors[key].load_state_dict(value_dict)
    else:
        raise ValueError(
            f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
        )

    # set correct dtype & device
    attn_processors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in attn_processors.items()}
    non_attn_lora_layers = [(t, l.to(device=self.device, dtype=self.dtype)) for t, l in non_attn_lora_layers]

    # set layers
    self.set_attn_processor(attn_processors)

    # set ff layers
    for target_module, lora_layer in non_attn_lora_layers:
        target_module.set_lora_layer(lora_layer)
