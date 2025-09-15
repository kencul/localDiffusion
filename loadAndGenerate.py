import torch
import os
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# --- 1. Define Paths ---
local_model_dir = "./flux_schnell_quantized" # Make sure this path is correct
transformer_path = os.path.join(local_model_dir, "transformer")
text_encoder_2_path = os.path.join(local_model_dir, "text_encoder_2")
vae_path = os.path.join(local_model_dir, "vae")

dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-schnell"
revision = "refs/pr/1"

# --- 2. Load All Models ---
print("Loading models into system RAM...")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", revision=revision)

# MODIFICATION: Force models to load fully on CPU to avoid meta device errors
transformer = FluxTransformer2DModel.from_pretrained(transformer_path, torch_dtype=dtype, low_cpu_mem_usage=False)
text_encoder_2 = T5EncoderModel.from_pretrained(text_encoder_2_path, torch_dtype=dtype, low_cpu_mem_usage=False)
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype, low_cpu_mem_usage=False)
print("✅ Models loaded into RAM.")

# --- 3. Build and Move Pipeline ---
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

print("Moving pipeline to GPU... (This may take several minutes)")
pipe.to("cuda")
print("✅ Pipeline on GPU.")

# --- 4. Generate Image ---
generator = torch.Generator(device='cuda').manual_seed(12345)

image = pipe(
    prompt='nekomusume cat girl, digital painting',
    num_inference_steps=4,
    generator=generator,
).images[0]

image.save('test_flux_fast_load.png')
print("Image saved!")