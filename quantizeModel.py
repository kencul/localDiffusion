import torch
import os
import gc
import json
from optimum.quanto import freeze, qfloat8, qint4, quantize, quantization_map
from safetensors.torch import save_file
from diffusers import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import T5EncoderModel
import transformers
import logging

def flush():
    gc.collect()
    torch.cuda.empty_cache()

flush()

# Set the library to show every detail of what it is doing
transformers.utils.logging.set_verbosity_debug()
# Also enable standard python logging to see background threads
logging.basicConfig(level=logging.DEBUG)

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
# vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision, device_map={"": "cpu"}, low_cpu_mem_usage=True)
# text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision, device_map="auto", low_cpu_mem_usage=True)

# --- 3. Quantize Models (This is the slow part) ---
print("Quantizing transformer to qint4...")
quantize(transformer, weights=qint4, exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
freeze(transformer)

# print("Quantizing text_encoder_2 to qint4...")
# quantize(text_encoder_2, weights=qint4)
# freeze(text_encoder_2)

# print("Quantizing vae to qfloat8...")
# quantize(vae, weights=qfloat8)
# freeze(vae)

# --- 4. Save the Quantized Models to Disk ---
# ------------------
# Transformer

# Save the config file (instant)
transformer.save_config(transformer_path)

# Save the Quantization Map (Required to reload correctly)
qmap = quantization_map(transformer)
with open(os.path.join(output_dir, "quantization_map.json"), "w") as f:
    json.dump(qmap, f)

# Save the Weights (Bypasses the 'save_pretrained' hang)
# save as one large file to avoid the sharding deadlock
save_file(transformer.state_dict(), os.path.join(output_dir, "model.safetensors"))

print(f"\nSuccess! Transformer saved to {output_dir}")
flush()


# Text Encoder
# print(f"Saving text_encoder_2 to {text_encoder_2_path}...")
# text_encoder_2.save_pretrained(text_encoder_2_path, max_shard_size ="2GB", safe_serialization=True)

# VAE
# print(f"Saving vae to {vae_path}...")
# vae.save_pretrained(vae_path)

print("All quantized models have been saved!")