import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.flow_utils import RAFT
from segment_anything import SamPredictor, sam_model_registry
from torch.nn.functional import l1_loss
from checkpoints.dino import vision_transformer
from checkpoints.dino.utils import load_pretrained_weights


class DenseMotionAlignmentLoss(nn.Module):
    def __init__(self):
        super(DenseMotionAlignmentLoss, self).__init__()

        # make flow network
        self.flow_net = RAFT()

        # make sam model
        sam_checkpoint = "checkpoints/sam/sam_vit_b_01ec64.pth"
        # model_type = "vit_h"
        model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(sam)
        # make dino model
        dino_architecture = "vit_small"
        dino_patch_size = 8
        self.dino = vision_transformer.__dict__[dino_architecture](patch_size=dino_patch_size,
                                                                   num_classes=0)
        load_pretrained_weights(self.dino, "checkpoints/dino/dino_deitsmall8_pretrain.pth", "teacher",
                                dino_architecture, dino_patch_size)

    def _calc_pixel_semantic_embedding(self, image):
        # image.shape =  (B, C, H, W)
        # B bs, H ih, W iw
        bs, _, ih, iw = image.shape
        # we return the output tokens from the `n` last blocks
        patch_semantic_embedding = self.dino.get_intermediate_layers(image, n=1)[0]
        # we discard the [CLS] token
        patch_semantic_embedding = patch_semantic_embedding[:, 1:, :]
        dino_patch_size = self.dino.patch_embed.patch_size
        eh, ew = int(ih / dino_patch_size), int(iw / dino_patch_size)
        # embedding dim = 384
        embedding_dim = patch_semantic_embedding.shape[-1]
        patch_semantic_embedding = patch_semantic_embedding.reshape(bs, eh, ew, embedding_dim)
        patch_semantic_embedding = patch_semantic_embedding.permute(0, 3, 1, 2)
        # bicubic, high image quality
        pixel_semantic_embedding = F.interpolate(patch_semantic_embedding, size=(ih, iw),
                                                 mode="bicubic", align_corners=True)
        # pixel_semantic_embedding.shape = (B, C, H, W)
        return pixel_semantic_embedding

    def _image_warp(self, image, flow):
        _, _, h, w = flow.shape
        device = flow.device

        # Normalize flow
        """
        Normalize pixel-offset (relative) flow to absolute [-1, 1] flow 
        (B, 2, H, W) -> (B, H, W, 2) For F.grid_sample
        """
        # Get base pixel coordinates
        base = torch.meshgrid(torch.arange(h), torch.arange(w))[::-1]
        base = torch.stack(base).float().to(device)

        # Convert to absolute coordinates
        flow = flow + base

        # Convert to [-1, 1] for grid_sample
        size = torch.tensor([w, h]).float().to(device)
        flow = -1 + 2. * flow / (-1 + size)[:, None, None]
        flow = flow.permute(0, 2, 3, 1)

        # warp image
        warped = F.grid_sample(image, flow, padding_mode="reflection", align_corners=True)
        return warped

    def _charbonnier_loss(self, pred_value, target_value, alpha=0.45, epsilon=1e-3, foreground_mask=None):
        delta = target_value - pred_value
        # shape (B, C, H, W) -> (B, H, W)
        delta = torch.mean(delta, dim=1)
        charbonnier_loss = torch.pow(torch.square(delta) + epsilon ** 2, alpha)
        charbonnier_loss = charbonnier_loss.sum(dim=(1, 2))
        foreground_size = torch.sum(foreground_mask, dim=(1, 2))
        charbonnier_loss = charbonnier_loss / (foreground_size + 1e-8)
        return charbonnier_loss[0]

    @torch.no_grad()
    def _PTracing_image(self,
                        prev_recons_image,  # F0
                        curr_recons_image,  # F1
                        curr_hps,  # handle_points
                        prev_hps,  # handle_points_init
                        r_p=3):
        _, _, max_r, max_c = prev_recons_image.shape
        for i in range(len(curr_hps)):
            pi0, pi = prev_hps[i], curr_hps[i]
            # f0.shape = (B, C)
            f0 = prev_recons_image[:, :, int(pi0[0]), int(pi0[1])]
            r1, r2 = max(0, int(pi[0]) - r_p), min(max_r, int(pi[0]) + r_p + 1)
            c1, c2 = max(0, int(pi[1]) - r_p), min(max_c, int(pi[1]) + r_p + 1)
            # curr_recons_image_neighbor.shape = (B, C, 2 * r_p, 2* r_p)
            curr_recons_image_neighbor = curr_recons_image[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - curr_recons_image_neighbor).abs().sum(dim=1)
            # all_dist.shape = (2 * r_p, 2* r_p)
            all_dist = all_dist.squeeze(dim=0)
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            curr_hps[i][0] = r1 + row
            curr_hps[i][1] = c1 + col
        return curr_hps

    def forward(self, prev_recons_image, hps, tps, curr_recons_image):
        # challenge: only one drag point
        device = curr_recons_image.device
        dense_motion_alignment_loss = torch.tensor(0.).to(device)

        # Normalize [-1, 1] -> [0, 1]
        prev_recons_image_norm = prev_recons_image / 2. + 0.5
        curr_recons_image_norm = curr_recons_image / 2. + 0.5

        # (B, H, W, C)
        # prev_recons_image_segmentation = prev_recons_image_norm.cpu().permute(0, 2, 3, 1).numpy()[0]
        prev_recons_image_segmentation = prev_recons_image_norm.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        prev_recons_image_segmentation = (prev_recons_image_segmentation * 255).astype(np.uint8)
        self.sam_predictor.set_image(prev_recons_image_segmentation)

        for i in range(len(hps)):
            hp, tp = hps[i], tps[i]
            # (y, x); however, sam need (x, y)
            prev_hp_segmentation = hp[[1, 0]].detach().unsqueeze(dim=0).cpu().numpy()
            """
            prev_hp_segmentation = hp.detach().unsqueeze(dim=0).cpu().numpy()
            """
            label_segmentation = np.array([1])
            # mask.shape = (1, H, W)  <class 'numpy.ndarray'>
            prev_mask, prev_score, prev_logit = self.sam_predictor.predict(
                point_coords=prev_hp_segmentation,
                point_labels=label_segmentation,
                multimask_output=False,
            )
            prev_mask = torch.tensor(prev_mask).to(device)

            # drag points -> flow fields
            # Pixels closer to the drag point receive a greater influence
            # one point (y, x)
            sparse_optical_flows = tp - hp
            _, _, h, w = curr_recons_image.shape
            flow_yaxis, flow_xaxis = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            flow_yaxis = flow_yaxis.to(device)
            flow_xaxis = flow_xaxis.to(device)
            # 2D gaussian
            sigma = torch.tensor([25. * (h / 512.), 25. * (w / 512.)]).to(device)
            # gaussian_coefficient.shape = (h, w)
            gaussian_coefficient = torch.exp(
                -(((flow_yaxis - tp[0]) ** 2) / (2 * sigma[0] ** 2)
                + ((flow_xaxis - tp[1]) ** 2) / (2 * sigma[1] ** 2))
            )
            dense_optical_flow_y = (gaussian_coefficient * sparse_optical_flows[0])
            dense_optical_flow_x = (gaussian_coefficient * sparse_optical_flows[1])
            # dense_optical_flow.shape = (h, w, 2)
            dense_optical_flow = torch.stack((dense_optical_flow_y, dense_optical_flow_x), dim=2)
            # shape (h, w, 2) -> (bs 1, 2, h, w)
            dense_optical_flow = dense_optical_flow.unsqueeze(dim=0).permute(0, 3, 1, 2)

            # flow prediction
            # flow (prev coordinate)
            # warp(prev_recons_image, flow) = curr_recons_image
            # pred_optical_flow.shape = (1, 2, 512, 512)
            pred_optical_flow = self.flow_net(curr_recons_image_norm, prev_recons_image_norm)
            # shape (1, H, W) -> (1, 2, H, W)
            prev_mask_flow = prev_mask.unsqueeze(dim=1).expand(-1, 2, -1, -1)
            # The points on the object cannot represent the entity
            dense_optical_flow = dense_optical_flow * prev_mask_flow
            pred_optical_flow = pred_optical_flow * prev_mask_flow
            displacement_loss = l1_loss(dense_optical_flow, pred_optical_flow)

            # color + semantic similarity
            warp_prev_recons_image_norm = self._image_warp(prev_recons_image_norm, dense_optical_flow)
            pixel_appearance_loss = self._charbonnier_loss(curr_recons_image_norm, warp_prev_recons_image_norm,
                                                        foreground_mask=prev_mask)
            curr_semantic_embedding = self._calc_pixel_semantic_embedding(curr_recons_image_norm)
            warp_semantic_embedding = self._calc_pixel_semantic_embedding(warp_prev_recons_image_norm)
            pixel_semantic_loss = self._charbonnier_loss(curr_semantic_embedding, warp_semantic_embedding,
                                                         foreground_mask=prev_mask)
            dense_motion_alignment_loss += (
                0.02 * displacement_loss + (1 - 0.02) * (pixel_appearance_loss + pixel_semantic_loss)
            ) / 3

        # handle_points_init = copy.deepcopy(handle_points)
        # curr_hps = hps.clone()
        curr_hps = [hp.clone() for hp in hps]
        curr_hps = self._PTracing_image(prev_recons_image_norm, curr_recons_image_norm, curr_hps, hps)

        dense_motion_alignment_loss = dense_motion_alignment_loss / len(hps)

        return dense_motion_alignment_loss, curr_recons_image, curr_hps