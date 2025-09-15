import torch
import os
from optimum.quanto import freeze, qfloat8, qint4, quantize
from diffusers import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import T5EncoderModel

# --- 1. Define Model and Output Paths ---
bfl_repo = "black-forest-labs/FLUX.1-schnell"
revision = "refs/pr/1"
dtype = torch.bfloat16

# This will be the main directory for your saved models
output_dir = "./flux_schnell_quantized" 
transformer_path = os.path.join(output_dir, "transformer")
text_encoder_2_path = os.path.join(output_dir, "text_encoder_2")
vae_path = os.path.join(output_dir, "vae")

# --- 2. Load Original Models ---
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)

# --- 3. Quantize Models (This is the slow part) ---
print("Quantizing transformer to qint4...")
quantize(transformer, weights=qint4, exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
freeze(transformer)

print("Quantizing text_encoder_2 to qint4...")
quantize(text_encoder_2, weights=qint4)
freeze(text_encoder_2)

print("Quantizing vae to qfloat8...")
quantize(vae, weights=qfloat8)
freeze(vae)

# --- 4. Save the Quantized Models to Disk ---
print(f"Saving transformer to {transformer_path}...")
transformer.save_pretrained(transformer_path)

print(f"Saving text_encoder_2 to {text_encoder_2_path}...")
text_encoder_2.save_pretrained(text_encoder_2_path)

print(f"Saving vae to {vae_path}...")
vae.save_pretrained(vae_path)

print("All quantized models have been saved!")