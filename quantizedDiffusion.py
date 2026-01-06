import torch
import time
from optimum.quanto import freeze, qfloat8, qint4, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# --- 1. Paths to your saved models ---
local_base = "./flux_schnell_quantized"
bfl_repo = "black-forest-labs/FLUX.1-schnell" # Still needed for tokenizers/schedulers
dtype = torch.bfloat16

print("Loading local quantized models...")

# --- 2. Load and Re-Quantize the Skeleton ---
# Note: We load the config from local, quantize the empty shell, THEN load the weights.

# Load Transformer
transformer = FluxTransformer2DModel.from_pretrained(f"{local_base}/transformer", torch_dtype=dtype)
quantize(transformer, weights=qint4, exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
freeze(transformer)

# Load Text Encoder 2 (T5)
text_encoder_2 = T5EncoderModel.from_pretrained(f"{local_base}/text_encoder_2", torch_dtype=dtype)
quantize(text_encoder_2, weights=qint4)
freeze(text_encoder_2)

# Load VAE
vae = AutoencoderKL.from_pretrained(f"{local_base}/vae", torch_dtype=dtype)
quantize(vae, weights=qfloat8)
freeze(vae)

# --- 3. Load remaining components from Hub (they are small/not quantized) ---
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2")

# --- 4. Assemble Pipeline ---
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

pipe.enable_model_cpu_offload()

# --- 5. Generate ---
prompt = "A futuristic city in the style of cyberpunk, 8k resolution"
image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    num_inference_steps=4, 
    guidance_scale=0.0, # Schnell usually uses 0 guidance
).images[0]

image.save("output_local.png")
print("Done!")