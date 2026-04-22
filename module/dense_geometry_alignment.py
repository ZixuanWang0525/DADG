import numpy as np
import os
import yaml
import pickle
import copy
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from typing import List, Dict, Any, Optional, Tuple
from utils.open_vocabulary_detection import GradientDetection

class DenseGeometryAlignmentLoss(nn.Module):
    def __init__(
        self,
        base_models = None,
        device: str = "cuda",
    ):
        super(DenseGeometryAlignmentLoss, self).__init__()

        self.device = device
        self.gradient_detector = base_models.gradient_detector

    def _boxes_to_masks(
        self,
        boxes: torch.Tensor,
        height: int = 64,
        width: int = 64,
    ) -> torch.Tensor:
        if boxes.dim() != 2 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [N,4]")

        boxes = boxes.float().clone()

        boxes[:, [0, 2]] = boxes[:, [0, 2]] * width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * height

        N = boxes.shape[0]
        masks = torch.zeros((N, height, width), dtype=torch.float32, device=boxes.device)

        for i in range(N):
            x1, y1, x2, y2 = boxes[i]

            x1 = int(torch.clamp(torch.floor(x1), 0, width - 1).item())
            y1 = int(torch.clamp(torch.floor(y1), 0, height - 1).item())
            x2 = int(torch.clamp(torch.ceil(x2), 0, width).item())
            y2 = int(torch.clamp(torch.ceil(y2), 0, height).item())

            if x2 <= x1 or y2 <= y1:
                continue

            masks[i, y1:y2, x1:x2] = 1.0

        return masks
    
    def _prepare_mask_gt(
        self,
        mask_gt,
        device,
        height: int = 64,
        width: int = 64,
        is_boxes: bool = False,
    ):
        if not is_boxes:
            # mask_gt (list)
            if isinstance(mask_gt[0], np.ndarray):
                return torch.tensor(mask_gt).to(device).float()
            elif isinstance(mask_gt[0], torch.Tensor):
                return torch.stack(mask_gt, dim=0).to(device).float()
        else:
            boxes = torch.tensor(mask_gt, dtype=torch.float32, device=device)
            return self._boxes_to_masks(
                boxes,
                height=height,
                width=width,
            )

    def _detection_with_gradient(
        self,
        recons_image: torch.Tensor,
        cls_gt: List[str],
    ):
        seg_out = self.gradient_detector(
            recons_image,
            class_names=cls_gt,
        )

        masks_pred = seg_out["masks"][0]     # [N,H,W]
        labels_pred = list(seg_out["labels"])     # list[str]

        masks_pred = F.interpolate(
            masks_pred.float().unsqueeze(1),   # [N,1,H,W]
            size=(64, 64),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)   # [N,64,64]

        masks_pred = torch.sigmoid(masks_pred)

        return {
                "masks": masks_pred,   # [N,64,64]
                "labels": labels_pred,
            }
    
    def _calc_size_proportion(self, foreground_mask):
        foreground_mask = foreground_mask.float()

        foreground_size = foreground_mask.sum(dim=(1, 2))  # [N]
        size_proportion = foreground_size.unsqueeze(dim=1) / (
            foreground_size.unsqueeze(dim=0) + 1e-8
        )
        return size_proportion
    
    def _calc_centroid(self, foreground_mask):
        eps = 1e-6
        h, w = foreground_mask.shape

        ys = torch.linspace(0.0, 1.0, h, device=foreground_mask.device, dtype=foreground_mask.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=foreground_mask.device, dtype=foreground_mask.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        mass = foreground_mask.sum() + eps
        cy = (foreground_mask * yy).sum() / mass
        cx = (foreground_mask * xx).sum() / mass

        return torch.stack([cy, cx], dim=0)
    
    def forward(self, recons_image, cls_gt, mask_gt, is_boxes=False):
        device = recons_image.device

        mask_gt = self._prepare_mask_gt(mask_gt, device, is_boxes=is_boxes)   # [N,64,64]

        dense_geometry_alignment_loss = torch.tensor(0.).to(device)
        _lambda = torch.tensor(0.25).to(device)

        seg_out = self._detection_with_gradient(recons_image=recons_image, cls_gt=cls_gt)

        mask_pred = seg_out["masks"]  # [N,64,64]
        cls_txt_pred = seg_out["labels"] # list[str]

        meaningful_mask_pred = []
        foreground_centroid_gt = []
        foreground_centroid_pred = []
        # according to the cls_txt_pred
        cvg_loss = torch.tensor(0.).to(device)
        for p in range(len(cls_gt)):
            if cls_gt[p] in cls_txt_pred:
                curidx = cls_txt_pred.index(cls_gt[p])
                meaningful_mask_pred.append(mask_pred[curidx])

                cur_mask_gt = mask_gt[p]
                # cur_mask_gt.shape = (64, 64)
                # cur_mask_gt = cur_mask_gt.to(device)
                eps = 1e-8
                intersection = (cur_mask_gt * mask_pred[curidx]).sum()
                union = cur_mask_gt.sum() + mask_pred[curidx].sum() - intersection
                bind_loss = 1.0 - (intersection / (union + eps))

                cur_foreground_centroid_gt = self._calc_centroid(cur_mask_gt)
                cur_foreground_centroid_pred = self._calc_centroid(mask_pred[curidx])

                foreground_centroid_gt.append(cur_foreground_centroid_gt)
                foreground_centroid_pred.append(cur_foreground_centroid_pred)

                separating_and_strengthening_loss = torch.tensor(0.0, device=device)
                for q in range(len(cls_gt)):
                    if p == q:
                        continue

                    other_mask_gt = mask_gt[q].float()

                    _intersection = (other_mask_gt * mask_pred[curidx]).sum()
                    _union = other_mask_gt.sum() + mask_pred[curidx].sum() - _intersection
                    separating_and_strengthening_loss += _intersection / (_union + eps)

                if len(cls_gt) > 1:
                    separating_and_strengthening_loss /= (len(cls_gt) - 1)

                cvg_loss += (bind_loss + separating_and_strengthening_loss) / 2.
                # class duplicate
                del cls_txt_pred[curidx]
                # mask_pred.shape = (number of instance, 64, 64)
                mask_pred = torch.cat([mask_pred[:curidx, :, :], mask_pred[curidx + 1:, :, :]], 0)
            else:
                continue
        if len(cls_gt) > 1:
            cvg_loss /= (len(cls_gt) - 1.)

        size_loss = torch.tensor(0.).to(device)
        distance_loss = torch.tensor(0.).to(device)

        if len(meaningful_mask_pred) <= 1:
            pass
        else:
            # foreground_centroid_gt.shape (number of instance, 2)
            # foreground_centroid_gt = torch.tensor(foreground_centroid_gt).to(device)
            foreground_centroid_gt = torch.stack(foreground_centroid_gt).to(device)
            # foreground_centroid_pred = torch.tensor(foreground_centroid_pred).to(device)
            foreground_centroid_pred = torch.stack(foreground_centroid_pred).to(device)
            if isinstance(meaningful_mask_pred[0], np.ndarray):
                meaningful_mask_pred = torch.tensor(meaningful_mask_pred).to(device)
            elif isinstance(meaningful_mask_pred[0], torch.Tensor):
                meaningful_mask_pred = torch.stack(meaningful_mask_pred, dim=0).to(device)

            #  size 
            size_gt = self._calc_size_proportion(mask_gt)
            size_pred = self._calc_size_proportion(meaningful_mask_pred)
            size_loss = F.mse_loss(size_pred, size_gt)

            #  distance
            dm_gt = foreground_centroid_gt[:, None, :] - foreground_centroid_gt[None, :, :]
            dm_pred = foreground_centroid_pred[:, None, :] - foreground_centroid_pred[None, :, :]
            """
            dm_gt = torch.sqrt(dm_gt[..., 0] ** 2 + dm_gt[..., 1] ** 2)
            dm_pred = torch.sqrt(dm_pred[..., 0] ** 2 + dm_pred[..., 1] ** 2)
            """
            dm_gt = dm_gt[..., 0] ** 2 + dm_gt[..., 1] ** 2
            dm_pred = dm_pred[..., 0] ** 2 + dm_pred[..., 1] ** 2
            # because diagonal element is 0
            diagonal_mask = torch.ones_like(dm_gt, dtype=torch.bool).to(device)
            torch.diagonal(diagonal_mask).fill_(0)
            dm_gt = dm_gt[diagonal_mask]
            dm_pred = dm_pred[diagonal_mask]
            distance_loss = F.mse_loss(dm_pred, dm_gt)

        dense_geometry_alignment_loss = (1 - _lambda) * cvg_loss + _lambda * (
            (distance_loss + size_loss) / 2)
        
        return dense_geometry_alignment_loss