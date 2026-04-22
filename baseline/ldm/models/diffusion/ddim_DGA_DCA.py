"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

#from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
#    extract_into_tensor
from baseline.ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSamplerWithGrad(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.model.module.num_timesteps, verbose=verbose)

        alphas_cumprod = self.model.module.alphas_cumprod
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.module.device)

        self.register_buffer('betas', to_torch(self.model.module.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.module.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample(self,
               batch_size,
               shape,
               num_ddim_steps=500,
               cond_embed=None,
               uncond_embed=None,
               eta=0.,
               CFG_scale=1.,
               num_recursive_steps=1,
               clip_grad=0,
               guidance_weight=1.0,
               guidance_schedule=None,
               cls_gt=None,
               mask_gt=None,
               language_prompt=None,
               DGALoss=None,
               DCALoss=None,
               mode="DCA_DGA",
            ):
        # Get schedule info
        self.make_schedule(ddim_num_steps=num_ddim_steps, ddim_eta=eta, verbose=False)
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        # Get steps info
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # Get shape and device info
        C, H, W = shape
        shape = (batch_size, C, H, W)
        device = self.model.module.betas.device
        b = batch_size

        # Make guidance schedule if we didn't get one
        if guidance_schedule is None:
            guidance_schedule = np.ones(500)
        
        guidance_step_numbers =  np.sum(guidance_schedule != 0)
        if mode == "DCA_DGA":
            dga_step_numbers = guidance_step_numbers * 0.8
        else:
            dga_step_numbers = guidance_step_numbers

        # Init noisy latent
        noisy_latent = torch.randn(shape, device=device)

        # Freeze model
        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        # Track stuff
        losses = []
        noise_norms = []
        guidance_norms = []

        # DDIM sampling loop
        for i, step in enumerate(iterator):
            # Bookkeeping
            index = total_steps - 1 - i
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            b, *_, device = *noisy_latent.shape, noisy_latent.device

            # Get variance schedule params
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            beta_t = a_t / a_prev

            # For each step, do N extra "recursive" steps
            subimages_entities = None
            subimages_relations = None
            for j in range(num_recursive_steps):

                # Set up grad (for differentiating guidance energy)
                torch.set_grad_enabled(True)
                noisy_latent_grad = noisy_latent.detach().requires_grad_(True)

                # Get CFG noise estimates
                x_in = torch.cat([noisy_latent_grad] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([uncond_embed, cond_embed])
                e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + CFG_scale * (e_t - e_t_uncond)

                # Get one step approximation
                pred_x0 = (noisy_latent_grad - sqrt_one_minus_at * e_t) / a_t.sqrt()
                recons_image = self.model.module.decode_first_stage_with_grad(pred_x0)

                # Get gradient of energy function
                # energy, info_loss = guidance_energy(recons_image, src_img)
                if mode == "DCA_DGA":
                    if i < dga_step_numbers:
                        energy = DGALoss(recons_image, cls_gt, mask_gt)
                    else:
                        recons_image = recons_image.squeeze(dim=0)
                        energy, subimages_entities, subimages_relations = DCALoss(
                            recons_image=recons_image,
                            language_prompt=language_prompt,
                            timestep=(i + 1),
                            self_recurrent=(j + 1),
                            subimages_entities=subimages_entities,
                            subimages_relations=subimages_relations
                        )
                        recons_image = recons_image.unsqueeze(dim=0)
                else:
                    energy = DGALoss(recons_image, cls_gt, mask_gt)
                if energy.item() != 0:
                    grad = torch.autograd.grad(energy, noisy_latent_grad)[0]
                else:
                    grad = torch.tensor(0.).to(device)
                grad = -grad * guidance_weight * guidance_schedule[i]
                # Clip gradient
                if clip_grad != 0:
                    grad_norm = torch.linalg.norm(sqrt_one_minus_at * grad.detach())
                    if grad_norm > clip_grad:
                        grad = grad / grad_norm * clip_grad

                # Update noise estimate with guidance gradiaent
                # TODO: This may be a bug, try not multiplying here. But need new hyperparams probs
                e_t = e_t - sqrt_one_minus_at * grad.detach()
                noisy_latent_grad = noisy_latent_grad.requires_grad_(False)

                # Logging
                losses.append(energy.item())
                noise_norms.append(torch.linalg.norm(e_t).item())
                guidance_norms.append(torch.linalg.norm(sqrt_one_minus_at * grad.detach()).item())

                del noisy_latent_grad, pred_x0, recons_image, grad, x_in

                torch.set_grad_enabled(False)

                # DDIM step
                with torch.no_grad():
                    # current prediction for x_0
                    pred_x0 = (noisy_latent - sqrt_one_minus_at * e_t) / a_t.sqrt()

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t

                    # random noise
                    noise = sigma_t * noise_like(noisy_latent.shape, device, False)

                    # DDIM step, get prev latent z_{t-1}
                    noisy_latent_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

                    # Inject noise (sample forward process) for recursive denoising
                    recur_noise = noise_like(noisy_latent.shape, device, False)
                    noisy_latent = beta_t.sqrt() * noisy_latent_prev + (1 - beta_t).sqrt() * recur_noise

                    del pred_x0, dir_xt, noise

            # When done making recursive steps, set current latent to t-1 latent
            noisy_latent = noisy_latent_prev

        # Make info dict to output
        info = {
            'losses': np.array(losses),
            'noise_norms': np.array(noise_norms),
            'guidance_norms': np.array(guidance_norms)
        }

        return noisy_latent, info
