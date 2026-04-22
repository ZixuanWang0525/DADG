from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR))

import logging
logging.disable(logging.CRITICAL)

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from PIL import Image
import numpy as np

import torch
from torchvision import utils
from torchvision.transforms.functional import to_tensor

from baseline.ldm.util import instantiate_from_config
from baseline.ldm.models.diffusion.ddim_DGA_DCA import DDIMSamplerWithGrad

import pickle
from torch import einsum
from einops import rearrange, repeat
from baseline.ldm.modules.attention import default, exists
from transformers import CLIPTokenizer

from data_loader import dense_diffusion_loader
from module.dense_geometry_alignment import DenseGeometryAlignmentLoss
from module.dense_concept_alignment import DenseConceptAlignmentLoss
from module.loading_base_model import Base_models

global sreg, creg, reg_sizes, COUNT, creg_maps, sreg_maps


def dense_diffusion_forward(self, x, context=None, mask=None):
    global sreg, creg, reg_sizes, COUNT, creg_maps, sreg_maps
    sa_ = True if context is None else False
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    # densediffusion
    # self + cross
    # 15 steps
    if COUNT / (32 * 2 * self.num_recursive_steps) < (self.timesteps * 0.3):
        current_ddim_step = COUNT // (32 * 2 * self.num_recursive_steps)
        current_ddim_step = min(current_ddim_step, self.timesteps - 1)

        treg = torch.pow(
            torch.tensor(
                (current_ddim_step * (1000 // self.timesteps)) / 1000,
                device=sim.device,
                dtype=sim.dtype
            ),
            5
        )

        # reg at self-attn
        if sa_:
            # 50% uncondition, 50% condition
            min_value = sim[int(sim.size(0) / 2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0) / 2):].max(-1)[0].unsqueeze(-1)
            sa_mask = sreg_maps[sim.size(1)].repeat(self.heads, 1, 1).to(sim.device, sim.dtype)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads, 1, 1).to(sim.device, sim.dtype)

            sim[int(sim.size(0) / 2):] += (sa_mask > 0) * size_reg * sreg * treg * (
                    max_value - sim[int(sim.size(0) / 2):])
            sim[int(sim.size(0) / 2):] -= ~(sa_mask > 0) * size_reg * sreg * treg * (
                    sim[int(sim.size(0) / 2):] - min_value)

        # reg at cross-attn
        else:
            min_value = sim[int(sim.size(0) / 2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0) / 2):].max(-1)[0].unsqueeze(-1)
            ca_mask = creg_maps[sim.size(1)].repeat(self.heads, 1, 1).to(sim.device, sim.dtype)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads, 1, 1).to(sim.device, sim.dtype)

            sim[int(sim.size(0) / 2):] += (ca_mask > 0) * size_reg * creg * treg * (
                    max_value - sim[int(sim.size(0) / 2):])
            sim[int(sim.size(0) / 2):] -= ~(ca_mask > 0) * size_reg * creg * treg * (
                    sim[int(sim.size(0) / 2):] - min_value)

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    COUNT += 1

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


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


# python -m baseline.DCA_DGA  --ddim_steps 50 --num_recursive_steps 10
# CUDA_VISIBLE_DEVICES=1 python -m baseline.DCA_DGA  --ddim_steps 50 --num_recursive_steps 1 --mode "DCA_DGA"
def main():
    parser = argparse.ArgumentParser()
    # Generation args
    parser.add_argument("--save_dir", default="./results/DCA_DGA_results", help='Path to save results')
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
    parser.add_argument("--guidance_weight", default=25.0, type=float)
    parser.add_argument("--num_recursive_steps", default=1, type=int)
    parser.add_argument("--guidance_schedule_path", type=str, default='baseline/guidance_schedule.npy',
                        help='use a custom guidance schedule')
    parser.add_argument("--clip_grad", type=float, default=200.0,
                        help='amount to clip guidance gradient by. 0.0 means no clipping')
    parser.add_argument("--seed", type=int, default=20241115, help="random seed")
    #
    parser.add_argument("--mode", type=str, default="DCA_DGA", help="")


    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    save_dir = Path(opt.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    # torch.save(opt, save_dir / 'config.pth')

    # Print for sanity check
    print(opt)

    # SETUP SAMPLING
    # Load model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    # Setup model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # DenseDiffusion
    module_count = 0
    for _module in model.modules():
        if _module.__class__.__name__ == "CrossAttention":
            _module.__class__.timesteps = opt.ddim_steps
            _module.__class__.num_recursive_steps = opt.num_recursive_steps
            _module.__class__.__call__ = dense_diffusion_forward
            module_count += 1

    print(f"[Info] Patched CrossAttention modules: {module_count}")

    # DataParallel
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
    DGALoss = DenseGeometryAlignmentLoss(base_models=base_models).cuda()
    DCALoss = DenseConceptAlignmentLoss(base_models=base_models).cuda()

    # BEGIN SAMPLING
    # prep data
    global sreg, creg
    sreg = .3
    creg = 1.
    # model params
    image_size = config["model"]["params"]["image_size"]
    # tokenizer
    clip_tokenizer = CLIPTokenizer.from_pretrained("./checkpoints/runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    # open data file
    with open('./dataset/DenseDiffusion/testset.pkl', 'rb') as f:
        seg_dataset = pickle.load(f)
    with open('./dataset/DenseDiffusion/testset_instances.pkl', 'rb') as f:
        inst_gt = pickle.load(f)
    layout_img_root = './dataset/DenseDiffusion/testset_layout/'
    error_data_idx = [57]
    for data_idx in range(len(seg_dataset)):
    # for data_idx in range(1):
        if data_idx in error_data_idx:
            continue
        global reg_sizes, creg_maps, sreg_maps
        cls_gt_, mask_gt_, reg_sizes_, sreg_maps_, \
        creg_maps_, text_cond_, language_prompt = dense_diffusion_loader(dataset=seg_dataset,
                                                                                    inst_gt=inst_gt,
                                                                                    idx=data_idx,
                                                                                    layout_img_root=layout_img_root,
                                                                                    diffusion_pipeline=model,
                                                                                    tokenizer=clip_tokenizer,
                                                                                    sp_sz=image_size)
        reg_sizes = reg_sizes_
        creg_maps = creg_maps_
        sreg_maps = sreg_maps_

        # Get prompt embeddings
        uncond_embed = text_cond_[0].unsqueeze(dim=0)
        cond_embed = text_cond_[1].unsqueeze(dim=0)

        # Make new directory for this sample
        sample_save_dir = save_dir / f'sample_{data_idx:03}'
        sample_save_dir.mkdir(exist_ok=True, parents=True)

        # Sample
        global COUNT
        COUNT = 0
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
            cls_gt=cls_gt_,
            mask_gt=mask_gt_,
            language_prompt=language_prompt,
            DGALoss=DGALoss,
            DCALoss=DCALoss,
            mode=opt.mode
        )

        # Decode sampled latent
        sample_img = model.module.decode_first_stage(sample)
        sample_img = torch.clamp((sample_img + 1.0) / 2.0, min=0.0, max=1.0)
        # Save useful unfo
        utils.save_image(sample_img, sample_save_dir / f'pred.png')


if __name__ == "__main__":
    main()