import torch
import time

from optimum.quanto import freeze, qfloat8, quantize

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

# --- Model Loading ---
print("Loading models, please wait...")
dtype = torch.bfloat16

bfl_repo = "black-forest-labs/FLUX.1-schnell"
revision = "refs/pr/1"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

# --- Quantization ---
print("Quantizing models...")
quantize(transformer, weights=qfloat8)
freeze(transformer)

quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

# --- Pipeline Setup ---
print("Setting up the pipeline...")
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
pipe.enable_model_cpu_offload()

generator = torch.Generator().manual_seed(12345)
image_counter = 0

print("\nReady for prompts!")

# --- Generation Loop ---
while True:
    prompt = input("Enter a prompt (or 'exit' to quit): ")
    
    if prompt.lower() == 'exit':
        print("Exiting program.")
        break
        
    if not prompt.strip():
        print("Prompt is empty, please try again.")
        continue

    print(f"Generating image for: '{prompt}'")
    start_time = time.time()
    
    try:
        image = pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=4,
            generator=generator,
            guidance_scale=3.5,
        ).images[0]
        
        end_time = time.time()
        
        filename = f"output_{image_counter}.png"
        image.save(filename)
        print(f"Image saved as {filename} in {end_time - start_time:.2f} seconds.")
        image_counter += 1
        
    except Exception as e:
        print(f"An error occurred during image generation: {e}")