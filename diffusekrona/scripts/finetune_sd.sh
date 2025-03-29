subjects="teapot" # Subject Name
# teapot subject images are available at dataset link provided in the README
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base" # Model card
export OUTPUT_DIR="../outputs/${subjects}" # Where to save the model
export INSTANCE_DIR="../data/${subjects}/input/" # Where the input data is stored


#------------------------------------------------------------------------------------
#                                    Hyperparameters
#------------------------------------------------------------------------------------
attn_update_unet="kqvo"
a1=64
a2=8
krona_unet_k_rank_a1=$a1 # k matrix factorization rank of a1
krona_unet_k_rank_a2=$a2 # k matrix factorization rank of a2
krona_unet_q_rank_a1=$a1 # q matrix factorization rank of a1
krona_unet_q_rank_a2=$a2 # q matrix factorization rank of a2
krona_unet_v_rank_a1=$a1 # v matrix factorization rank of a1
krona_unet_v_rank_a2=$a2 # v matrix factorization rank of a2
krona_unet_o_rank_a1=$a1 # out matrix factorization rank of a1
krona_unet_o_rank_a2=$a2 # out matrix factorization rank of a2

lr=1e-3
steps=500

accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of sks${subjects}" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$steps \
    --adapter_type="krona" \
    --seed="0" \
    --diffusion_model="base" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --attn_update_unet=$attn_update_unet \
    --krona_unet_k_rank_a1=$krona_unet_k_rank_a1 \
    --krona_unet_k_rank_a2=$krona_unet_k_rank_a2 \
    --krona_unet_q_rank_a1=$krona_unet_q_rank_a1 \
    --krona_unet_q_rank_a2=$krona_unet_q_rank_a2 \
    --krona_unet_v_rank_a1=$krona_unet_v_rank_a1 \
    --krona_unet_v_rank_a2=$krona_unet_v_rank_a2 \
    --krona_unet_o_rank_a1=$krona_unet_o_rank_a1 \
    --krona_unet_o_rank_a2=$krona_unet_o_rank_a2
