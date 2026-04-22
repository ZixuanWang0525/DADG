from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR))

import logging
logging.disable(logging.CRITICAL)

import os
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from PIL import Image
import numpy as np

import torch
from torchvision import utils
from torchvision.transforms.functional import to_tensor

from baseline.ldm.util import instantiate_from_config
from baseline.ldm.models.diffusion.ddim_DCA import DDIMSamplerWithGrad

import pickle
from torch import einsum
from einops import rearrange, repeat
from baseline.ldm.modules.attention import default, exists
from transformers import CLIPTokenizer

from data_loader import CocoCaptionDataset
from torch.utils.data import DataLoader
from module.dense_concept_alignment import DenseConceptAlignmentLoss
from module.loading_base_model import Base_models

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


# python -m baseline.DCA  --ddim_steps 50 --num_recursive_steps 2
# CUDA_VISIBLE_DEVICES=1 python -m baseline.DCA  --ddim_steps 50 --num_recursive_steps 2
def main():
    parser = argparse.ArgumentParser()
    # Generation args
    parser.add_argument("--log_freq", type=int, default=35, help='frequency to log info')

    # Vanilla diffusion args
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="number of ddim sampling steps. n.b. this is kind of hardcoded, so maybe don't change")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta, 0 => deterministic")
    parser.add_argument("--scale", type=float, default=7.5, help="CFG scale")
    """
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model")
    """
    parser.add_argument("--config", type=str, default="baseline/configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/v1-5-pruned-emaonly.ckpt",
                        help="path to checkpoint of model")

    # Guidance args
    parser.add_argument("--guidance_weight", default=60.0, type=float)
    parser.add_argument("--num_recursive_steps", default=1, type=int)
    parser.add_argument("--guidance_schedule_path", type=str, default='baseline/guidance_schedule.npy',
                        help='use a custom guidance schedule')
    parser.add_argument("--clip_grad", type=float, default=200.0,
                        help='amount to clip guidance gradient by. 0.0 means no clipping')
    parser.add_argument("--seed", type=int, default=20241115, help="random seed")

    opt = parser.parse_args()

    # Print for sanity check
    print(opt)

    # SETUP SAMPLING
    # Load model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    # Setup model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()

    # Get DDIM sampler + guidance info
    sampler = DDIMSamplerWithGrad(model)

    torch.set_grad_enabled(False)

    # Get guidance schedule
    if opt.guidance_schedule_path:
        guidance_schedule = np.load(opt.guidance_schedule_path)
        new_guidance_schedule_indices = np.linspace(0, len(guidance_schedule) - 1, opt.ddim_steps, dtype=int)
        guidance_schedule = guidance_schedule[new_guidance_schedule_indices]
    else:
        guidance_schedule = None

    # Make loss function
    base_models = Base_models(
        grounding_model_dir="./checkpoints/grounding_dino",
        sam_checkpoint_path="./checkpoints/sam/sam_vit_b_01ec64.pth",
        CLIP_dir="./checkpoints/openai/clip-vit-large-patch14",
        device="cuda",
    )
    DCALoss = DenseConceptAlignmentLoss(base_models=base_models).cuda()

    # BEGIN SAMPLING
    # prep data
    # model params
    original_image_size = config["model"]["params"]["image_size"] * (2 ** (4 - 1))
    coco_val = CocoCaptionDataset(image_yres=original_image_size, image_xres=original_image_size)
    coco_loader = DataLoader(coco_val, batch_size=1, shuffle=False, num_workers=0)
    save_dir = "./results/DCA_results"
    for data_idx, sample in enumerate(coco_loader):
        image, caption, image_idx = sample

        if len(caption) != 0:
            for i in range(len(caption)):
                image_caption = caption[i]["caption"]

                uncond_embed = model.module.get_learned_conditioning([""])
                cond_embed = model.module.get_learned_conditioning(image_caption)
                
                modify_image_caption = image_caption[0].replace(" ", "_")
                modify_image_caption = modify_image_caption.replace("/", "_")

                sample_save_dir = save_dir + "/" + f'{image_idx.item()}' + "_" + modify_image_caption + "/"
                if not os.path.exists(sample_save_dir):
                    os.makedirs(sample_save_dir)

                # Sample
                sample, info = sampler.sample(
                    num_ddim_steps=opt.ddim_steps,
                    cond_embed=cond_embed,
                    uncond_embed=uncond_embed,
                    batch_size=1,
                    shape=[4, 64, 64],
                    CFG_scale=opt.scale,
                    eta=opt.ddim_eta,
                    guidance_schedule=guidance_schedule,
                    num_recursive_steps=opt.num_recursive_steps,
                    clip_grad=opt.clip_grad,
                    guidance_weight=opt.guidance_weight,
                    log_freq=opt.log_freq,
                    results_folder=sample_save_dir,
                    language_prompt=image_caption[0],
                    guidance_energy=DCALoss
                )

                # Decode sampled latent
                sample_img = model.module.decode_first_stage(sample)
                sample_img = torch.clamp((sample_img + 1.0) / 2.0, min=0.0, max=1.0)

                # Save useful unfo
                utils.save_image(sample_img, sample_save_dir + modify_image_caption + ".png")
        print(f'Finish Sampling {data_idx} of {len(coco_loader)}')


if __name__ == "__main__":
    main()