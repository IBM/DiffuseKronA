export adapter_type="krona"
export attn_update_unet="kqvo"
export prompt="A sksdog6 op top of sofa"
export checkpoint_path="../outputs/dog6/krona_k64:8q64:8v64:8o64:8_sdxl_0.001/"

# dog6 subject images are available at dataset link provided in the README
accelerate launch inference_sdxl.py \
    --checkpoint_path=$checkpoint_path \
    --output_path=$checkpoint_path \
    --adapter_type=$adapter_type \
    --attn_update_unet=$attn_update_unet \
    --prompt "$prompt" \
    --seed=0
