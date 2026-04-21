import os, gc
import cv2
import torch
from torch import nn
import argparse
import numpy as np
import sys
sys.path.append('../')
import omniconfig
from deepcompressor.utils import tools
import json

from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, patchModel
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.app.diffusion.dataset.collect.online_infinity_generation import args_2b, args_8b


from Infinity_rep.infinity.models.infinity import Infinity
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_tokenizer, gen_one_img, load_transformer
from Infinity_rep.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from PIL import Image

from evaluation.build_functions import assemble_model, attach_kv_qparams

# --- HELPER FOR BOOLEAN ARGUMENTS ---
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class InfinityPipelineWrapper:
    def __init__(self, model, vae, tokenizer, text_encoder, generation_args):
        self.quantized_model = model
        self.vae = vae
        self.text_tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.generation_args = generation_args
        self.scheduler = 'Custom' # Dummy attribute for compatibility

    def to(self, device):
        self.quantized_model.to(device)
        return self

    def set_progress_bar_config(self, **kwargs):
        # This is a dummy method to satisfy the evaluation framework's API.
        # It doesn't need to do anything.
        pass

    def __call__(self, prompt: list[str], generator: list[torch.Generator], **kwargs):
        # The __call__ should handle a batch of prompts
        pil_images = []
        for i, p in enumerate(prompt):
            # Update the seed for each image in the batch
            self.generation_args['g_seed'] = generator[i].initial_seed()

            # Generate one image
            image_tensor = gen_one_img(
                self.quantized_model, self.vae, self.text_tokenizer,
                self.text_encoder, p, **self.generation_args
            )
            # Convert to the expected PIL format
            img = cv2.cvtColor(image_tensor.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img))
            # --- ADD THESE LINES TO FORCE MEMORY CLEANUP ---
            gc.collect()
            torch.cuda.empty_cache()

        # The framework expects an object with an 'images' attribute
        class PipelineOutput:
            def __init__(self, images):
                self.images = images

        return PipelineOutput(pil_images)

# Setup arguments
logger = tools.logging.getLogger(__name__)
ptq_config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
ptq_config.output.lock()

# We parse the 'unknown_args' left over from the config parser
parser = argparse.ArgumentParser()
parser.add_argument("--enable_weight_quant", type=str2bool, default=None, help="Enable weight quantization")
parser.add_argument("--enable_activation_quant", type=str2bool, default=None, help="Enable activation quantization")
parser.add_argument("--enable_kv_quant", type=str2bool, default=None, help="Enable KV cache quantization")
parser.add_argument("--gen-root", dest="gen_root", type=str, default=None, help="Override generation root path")
parser.add_argument("--ref-root", dest="ref_root", type=str, default=None, help="Override reference root path")
parser.add_argument("--base-path", dest="base_path", type=str, default=None, help="Override base path for model artifacts")

# Parse only the remaining arguments
custom_args, _ = parser.parse_known_args(unknown_args)

if ptq_config.pipeline.name == 'infinity_2b':
    args = args_2b
    cache_path = os.path.join('runs/', "kv_scales", "kv_quant_calib.pt")
elif ptq_config.pipeline.name == 'infinity_8b':
    args = args_8b
    cache_path = os.path.join('runs/', "kv_scales", "kv_quant_calib_8b.pt")
else:
    raise NotImplementedError(f"Pipeline {ptq_config.pipeline.name} not implemented")
print('Args:', args)

#  Extract the config objects as before
quant_config = ptq_config.quant
eval_config = ptq_config.eval

# --- OVERRIDE CONFIGS IF ARGUMENTS PROVIDED ---
if custom_args.gen_root is not None:
    print(f"Overriding gen_root: {custom_args.gen_root}")
    eval_config.gen_root = custom_args.gen_root

if custom_args.ref_root is not None:
    print(f"Overriding ref_root: {custom_args.ref_root}")
    eval_config.ref_root = custom_args.ref_root
# -----------------------------------------------
# Load base models
vae = load_visual_tokenizer(args)
model = load_transformer(vae, args)
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

if args.bf16:
    # --- CAST TO BFLOAT16 ---
    print("--- Casting VAE and Text Encoder to BFloat16 ---")
    vae = vae.to(dtype=torch.bfloat16)
    text_encoder = text_encoder.to(dtype=torch.bfloat16)
    # ---------------------------------

# Setup generation schedule
h_div_w = 1/1
h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

# --- QUANTIZATION FLAGS SETUP ---
# Default values
enable_weight_quant = True
enable_activation_quant = True
enable_kv_quant = True
skip_ca_kv_act = True

# Override defaults if command line args are present
if custom_args.enable_weight_quant is not None:
    enable_weight_quant = custom_args.enable_weight_quant
    print(f"Overriding enable_weight_quant: {enable_weight_quant}")

if custom_args.enable_activation_quant is not None:
    enable_activation_quant = custom_args.enable_activation_quant
    print(f"Overriding enable_activation_quant: {enable_activation_quant}")

if custom_args.enable_kv_quant is not None:
    enable_kv_quant = custom_args.enable_kv_quant
    print(f"Overriding enable_kv_quant: {enable_kv_quant}")


print("--- Patching attention layers to be compatible ---")
quantized_model = patchModel(model)
print("Patching complete.\n")


# 1. Load the saved artifacts for inference
print("--- Loading inference artifacts (model.pt, branch.pt) ---")
dtype = torch.bfloat16
# --- BASE PATH SETUP ---
base_path = 'runs/diffusion/int4_rank32_naive_2b/' # Default
if custom_args.base_path is not None:
    base_path = custom_args.base_path
    print(f"Overriding base_path: {base_path}")

print(f"Using base_path: {base_path}")
# -----------------------

print("  > Loading weights to CPU...")
weights = torch.load(os.path.join(base_path, 'model.pt'), map_location='cpu')

# LOAD SCALES/BRANCHES DIRECTLY TO GPU (Safe & Compatible)
try:
    # REMOVE map_location='cpu' here
    smooth_scales = torch.load(os.path.join(base_path, 'smooth.pt'))
except:
    smooth_scales = {}
    print('No Smoothing applied')

branch_path = os.path.join(base_path, 'branch.pt')
if os.path.isfile(branch_path):
    # REMOVE map_location='cpu' here
    branch_state_dict = torch.load(branch_path)
else:
    branch_state_dict = {}
    print('No Low Rank Branch Applied')

# 2. Crete the model Structure
model_struct = InfinityStruct.construct(quantized_model)
for name, module in quantized_model.named_modules():
    module.name = name

generation_args = { 'cfg_list': [3.0]*13, 'tau_list': [0.5]*13, 'g_seed': 16,
                    'gt_leak': 0, 'gt_ls_Bl': None, 'scale_schedule': scale_schedule,
                    'cfg_insertion_layer': [args.cfg_insertion_layer], 'vae_type': args.vae_type,
                    'sampling_per_bits': args.sampling_per_bits, 'enable_positive_prompt': False }

if enable_weight_quant:
    model_struct = assemble_model(model_struct = model_struct,
                                configs = ptq_config,
                                branch_state_dict = branch_state_dict,
                                smooth_scales = smooth_scales,
                                weights = weights,
                                quantize_activations = enable_activation_quant,
                                skip_ca_kv_act = skip_ca_kv_act)

if enable_kv_quant:
    attach_kv_qparams(quantized_model, cache_path)

del weights
del branch_state_dict
del smooth_scales
gc.collect()
torch.cuda.empty_cache()

infinity_pipeline = InfinityPipelineWrapper(quantized_model, vae, text_tokenizer, text_encoder, generation_args)

print(f"--- LAUNCHING EVALUATION ---")
print(f"  > Generated Images Path: {eval_config.gen_root}")
print(f"  > Reference Images Path: {eval_config.ref_root}")

results = eval_config.evaluate(infinity_pipeline, skip_gen=False, task=ptq_config.pipeline.task)

logger.info(f"* Saving results to {ptq_config.output.job_dirpath}")
with open(f"{eval_config.gen_root}/results.json", "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)
