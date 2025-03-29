import diffusers
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch, os, argparse
from load_attn_procs import load_attn_procs

def generator(checkpoint_path, output_dir, prompt, seed=0):
    diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs = load_attn_procs
    # create the image folder
    image_dir = os.path.join(output_dir, 'images') 
    if(os.path.exists(image_dir)): pass
    else: os.mkdir(image_dir)

    # load the SDXL model
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(checkpoint_path)
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    refiner.to("cuda"); generator = torch.Generator("cuda").manual_seed(seed)
    
    # generate images
    image = pipe(prompt=prompt, output_type="latent", generator=generator).images[0]
    image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]
        
    image_save_path = os.path.join(image_dir, f"image_{seed}.jpg")
    image.save(image_save_path)
    print(f"Image generation completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="path to the output folder")
    parser.add_argument("--prompt", type=str, required=True, help="prompt for the image generation")
    parser.add_argument("--seed", type=int, default=0, help="seed for the image generation")
    parser.add_argument("--adapter_type", type=str, default="krona", help="adapter type")
    parser.add_argument("--attn_update_unet", type=str, default="kqvo", help="attention update type")
    args = parser.parse_args()

    # make global variables
    os.environ["attn_update_unet"] = args.attn_update_unet
    os.environ["adapter_type"] = args.adapter_type

    # run the generator
    generator(args.checkpoint_path, args.output_path, args.prompt, args.seed)


if __name__ == "__main__":
    main()