import os
from typing import List, Dict, Any, Optional, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

class GroundingSAM(nn.Module):
    def __init__(
        self,
        grounding_model_dir: str,
        sam_checkpoint_path: str,
        sam_type: str = "vit_b",
        device: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        super(GroundingSAM, self).__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        if not os.path.exists(grounding_model_dir):
            raise FileNotFoundError(f"grounding_model_dir not found: {grounding_model_dir}")
        if not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"sam checkpoint not found: {sam_checkpoint_path}")
        # Grounding DINO
        self.processor = AutoProcessor.from_pretrained(grounding_model_dir)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model_dir
        ).to(self.device)
        self.grounding_model.eval()

        # SAM
        self.sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path).to(self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
    
    def load_image(self, image_path: str):
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        return image_pil, image_np
    
    @torch.no_grad()
    def detect_boxes(self, image_pil: Image.Image, text_prompt: str) -> Dict[str, Any]:
        inputs = self.processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.grounding_model(**inputs)

        results = self._process_results_dynamically(outputs, inputs.input_ids, image_pil.size[::-1])

        boxes = results["boxes"].detach().cpu().numpy() if len(results["boxes"]) > 0 else np.zeros((0, 4))
        scores = results["scores"].detach().cpu().numpy() if len(results["scores"]) > 0 else np.array([])
        labels = list(results["labels"])

        if len(results["boxes"]) == 0:
            return self._empty_detection_result()

        return {
            "boxes": boxes,     # [N, 4],
            "scores": scores,   # [N]
            "labels": labels,   # list[str]
        }
    
    @torch.no_grad()
    def detect_boxes_batch(self, image_pil: Image.Image, text_prompt: str) -> Dict[str, Any]:
        class_names = []

        for part in text_prompt.strip().split("."):
            name = part.strip().lower()
            if not name:
                continue
            else:
                class_names.append(name)

        batch_boxes = []
        batch_scores = []
        batch_labels = []

        for class_name in class_names:
            det = self.detect_boxes(image_pil, class_name)

            if len(det["boxes"]) > 0:
                best_idx = int(np.argmax(det["scores"]))
                batch_boxes.append(det["boxes"][best_idx])
                batch_scores.append(det["scores"][best_idx])
                batch_labels.append(class_name)

        if len(batch_boxes) == 0:
            return self._empty_detection_result()

        return {
            "boxes": np.stack(batch_boxes, axis=0),
            "scores": np.array(batch_scores),
            "labels": batch_labels
        }

    def _process_results_dynamically(self, outputs: Dict[str, Any], input_ids: torch.Tensor, 
            image_size: Tuple) -> Dict[str, Any]:
        box_threshold = self.box_threshold
        text_threshold = self.text_threshold

        for box_thr in self._generate_dynamic_thresholds(box_threshold):
            for text_thr in self._generate_dynamic_thresholds(text_threshold):
                result = self._post_process_with_thresholds(outputs, input_ids, image_size, box_thr, text_thr)
                if len(result["boxes"]) > 0:  
                    return result

        return self._empty_detection_result()

    def _generate_dynamic_thresholds(self, start: float, stop: float = 0.05, step: float = 0.05) -> Tuple[float, ...]:
        thresholds = []
        current = start

        while current >= stop:
            thresholds.append(round(current, 4))
            current -= step

        if not thresholds or thresholds[-1] != stop:
            thresholds.append(stop)

        return tuple(thresholds)

    def _post_process_with_thresholds(self, outputs: Dict[str, Any], input_ids: torch.Tensor, image_size: Tuple, 
            box_threshold: float, text_threshold: float) -> Dict[str, Any]:
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_size],  # (H, W)
        )[0]
        return results

    @staticmethod
    def _empty_detection_result() -> Dict[str, Any]:
        return {
            "boxes": np.zeros((0, 4)),
            "scores": np.array([]),
            "labels": [],
            "low_confidence_fallback": True,
        }
    
    @torch.no_grad()
    def segment_from_boxes(self, image_np: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        self.predictor.set_image(image_np)

        if len(boxes) == 0:
            h, w = image_np.shape[:2]
            return np.zeros((0, h, w), dtype=bool)

        masks = []
        for box in boxes:
            pred_masks, pred_scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            masks.append(pred_masks[0])

        return np.stack(masks, axis=0).astype(bool)
    
    @torch.no_grad()
    def predict(self, image_path: str, text_prompt: str, mode: str = 'segmentation') -> Dict[str, Any]:
        image_pil, image_np = self.load_image(image_path)
        detection_results = self.detect_boxes_batch(image_pil, text_prompt)

        if mode == 'detection':
            return {
                "boxes": detection_results["boxes"],
                "scores": detection_results["scores"],
                "labels": detection_results["labels"]
            }
        elif mode == 'segmentation':
            masks = self.segment_from_boxes(image_np, detection_results["boxes"])
            return {
                "boxes": detection_results["boxes"],
                "scores": detection_results["scores"],
                "labels": detection_results["labels"],
                "masks": masks  # [N, H, W]
            }
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose either 'detection' or 'segmentation'.")

       
class GradientDetection(nn.Module):
    
    def __init__(
        self,
        grounding_model_dir: str,
        sam_checkpoint_path: str,
        device: Optional[str] = None,
        image_size: int = 512,
        query_temperature: float = 0.07,
        sam_model_type: str = "vit_b"   
    ):
        super(GradientDetection, self).__init__()
        if not os.path.exists(grounding_model_dir):
            raise FileNotFoundError(f"grounding_model_dir not found: {grounding_model_dir}")
        if not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"sam checkpoint not found: {sam_checkpoint_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.query_temperature = query_temperature
        self.processor = AutoProcessor.from_pretrained(grounding_model_dir)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model_dir
        ).to(self.device)
        self.grounding_model.eval()

        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        sam.to(self.device)
        sam.train()
        self.sam = sam
        for p in self.sam.image_encoder.parameters():
                p.requires_grad = False

        self.resize_longest_side = ResizeLongestSide(self.sam.image_encoder.img_size)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.min() < 0:
            image = (image + 1.0) / 2.0
        return image.clamp(0.0, 1.0)
    
    def _normalize_image_dino(self, image: torch.Tensor) -> torch.Tensor:
        image = self._normalize_image(image)
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        pixel_mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=image.dtype,
            device=image.device,
        ).view(1, 3, 1, 1)

        pixel_std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=image.dtype,
            device=image.device,
        ).view(1, 3, 1, 1)

        image = (image - pixel_mean) / pixel_std
        return image

    def _encode_text_dino(self, class_name: str):
        text_inputs = self.processor(
            text=class_name,
            return_tensors="pt"
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        return text_inputs
    
    def _soft_select_box(
        self,
        pred_boxes: torch.Tensor,   # [B, Q, 4], normalized cxcywh
        logits: torch.Tensor        # [B, Q, T]
    ):
        query_scores = logits.sigmoid().amax(dim=-1)  # [B, Q]

        weights = F.softmax(query_scores / self.query_temperature, dim=-1)  # [B, Q]

        # aggregate
        soft_box_cxcywh = (weights.unsqueeze(-1) * pred_boxes).sum(dim=1)  # [B, 4]
        soft_score = (weights * query_scores).sum(dim=1)                    # [B]

        return soft_box_cxcywh, soft_score
    
    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """
        boxes: [..., 4], normalized cxcywh
        """
        cx, cy, w, h = boxes.unbind(dim=-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def detection_forward(self, image: torch.Tensor, class_names: List[str]) -> Dict[str, Any]:
        """
        image: [B, 3, H, W]
        class_names: ['cat', 'dog', ...]
        """
        assert image.dim() == 4, "image must be [B,3,H,W]"
        B, _, H, W = image.shape
        image = image.to(self.device)

        pixel_values = self._normalize_image_dino(image)

        batch_boxes = []
        batch_scores = []

        for class_name in class_names:
            text_inputs = self._encode_text_dino(class_name)
            #
            self.grounding_model.to(pixel_values.device)
            outputs = self.grounding_model(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask", None),
            )

            # HF GroundingDINO return:
            # outputs.pred_boxes: [B, Q, 4] normalized cxcywh
            # outputs.logits:     [B, Q, T]
            pred_boxes = outputs.pred_boxes
            logits = outputs.logits

            soft_box_cxcywh, soft_score = self._soft_select_box(pred_boxes, logits)
            soft_box_xyxy = self._cxcywh_to_xyxy(soft_box_cxcywh).clamp(0.0, 1.0)

            scale = torch.tensor([W, H, W, H], device=self.device, dtype=soft_box_xyxy.dtype)
            soft_box_xyxy = soft_box_xyxy * scale.unsqueeze(0)

            batch_boxes.append(soft_box_xyxy)   # [B,4]
            batch_scores.append(soft_score)     # [B]

        boxes_xyxy = torch.stack(batch_boxes, dim=1)   # [B,N,4]
        scores = torch.stack(batch_scores, dim=1)      # [B,N]

        return {
            "boxes": boxes_xyxy,
            "scores": scores,
            "labels": class_names,
        }
    
    @property
    def sam_img_size(self):
        return self.sam.image_encoder.img_size
    
    def _normalize_image_sam(self, image: torch.Tensor) -> torch.Tensor:
        image = self._normalize_image(image)

        b, c, h, w = image.shape

        target_size = self.resize_longest_side.get_preprocess_shape(h, w, self.sam_img_size)
        image = F.interpolate(
            image,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        pad_h = self.sam_img_size - image.shape[-2]
        pad_w = self.sam_img_size - image.shape[-1]
        image = F.pad(image, (0, pad_w, 0, pad_h))

        # image = (image - self.sam.pixel_mean[None, :, None, None]) / self.sam.pixel_std[None, :, None, None]
        image = (image - self.sam.pixel_mean) / self.sam.pixel_std
        return image
    
    def encode_image_sam(self, image: torch.Tensor):
        """
        image: [B,3,H,W] in [0,1]
        """
        b, _, h, w = image.shape
        input_image = self._normalize_image_sam(image)
        image_embeddings = self.sam.image_encoder(input_image)
        return image_embeddings, (h, w)
    
    def transform_boxes_from_detection(
        self,
        boxes_xyxy: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        h, w = original_size

        single_box_per_image = (boxes_xyxy.dim() == 2)
        if single_box_per_image:
            boxes_xyxy = boxes_xyxy.unsqueeze(1)   # [B,1,4]

        # SAM: keep aspect ratio, resize longest side to self.img_size
        new_h, new_w = self.resize_longest_side.get_preprocess_shape(h, w, self.sam_img_size)

        scale_x = boxes_xyxy.new_tensor(new_w / w)
        scale_y = boxes_xyxy.new_tensor(new_h / h)

        # keep soft box gradient
        scales = boxes_xyxy.new_tensor([scale_x, scale_y, scale_x, scale_y]).view(1, 1, 4)
        transformed_boxes = boxes_xyxy * scales

        """
        transformed_boxes[..., 0::2] = transformed_boxes[..., 0::2].clamp(0, float(new_w))
        transformed_boxes[..., 1::2] = transformed_boxes[..., 1::2].clamp(0, float(new_h))
        """
        x1 = transformed_boxes[..., 0].clamp(0, float(new_w))
        y1 = transformed_boxes[..., 1].clamp(0, float(new_h))
        x2 = transformed_boxes[..., 2].clamp(0, float(new_w))
        y2 = transformed_boxes[..., 3].clamp(0, float(new_h))

        transformed_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        if single_box_per_image:
            transformed_boxes = transformed_boxes[:, 0, :]   # [B,4]

        return transformed_boxes
    
    def decode_sam(
        self,
        image_embeddings: torch.Tensor,
        original_size,
        boxes_xyxy: Optional[torch.Tensor] = None,     # [B,4] or [B,N,4]
    ):
        device = image_embeddings.device

        boxes = None
        if boxes_xyxy is not None:
            boxes = self.transform_boxes_from_detection(boxes_xyxy, original_size).to(device)
            if boxes.dim() == 3 and boxes.shape[1] == 1:
                boxes = boxes[:, 0, :]

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=self.resize_longest_side.get_preprocess_shape(original_size[0], original_size[1], self.sam_img_size),
            original_size=original_size,
        )

        return {
            "masks": masks,              # logits if return_logits=True
            "low_res_masks": low_res_masks,    # decoder logits
            "iou_predictions": iou_predictions,
        }
    
    def segmentation_forward(
        self,
        image: torch.Tensor,
        boxes_xyxy: Optional[torch.Tensor] = None, 
    ):
        image_embeddings, original_size = self.encode_image_sam(image)
        return self.decode_sam(
            image_embeddings=image_embeddings,
            original_size=original_size,
            boxes_xyxy=boxes_xyxy,
        )
    
    def forward(
        self,
        image: torch.Tensor,
        class_names: List[str],
        mode: str = "segmentation",   # "detection" or "segmentation"
    ) -> Dict[str, Any]:
        image = image.to(self.device)
        image = image.float()

        det_out = self.detection_forward(image, class_names)

        if mode == "detection":
            return det_out

        boxes_all = det_out["boxes"]    # [B,N,4]
        B, N, _ = boxes_all.shape

        image_embeddings, original_size = self.encode_image_sam(image)

        batch_masks = []
        batch_low_res_masks = []
        batch_iou_predictions = []

        # one class / box,  call SAM
        for i in range(N):
            boxes_i = boxes_all[:, i, :]   # [B,4]

            seg_out = self.decode_sam(
                image_embeddings=image_embeddings,
                original_size=original_size,
                boxes_xyxy=boxes_i,
            )

            # seg_out["masks"]: [B,1,H,W]
            batch_masks.append(seg_out["masks"])
            batch_low_res_masks.append(seg_out["low_res_masks"])
            batch_iou_predictions.append(seg_out["iou_predictions"])

        masks = torch.stack(batch_masks, dim=1)                # [B,N,1,H,W]
        low_res_masks = torch.stack(batch_low_res_masks, dim=1)  # [B,N,1,h,w]
        iou_predictions = torch.stack(batch_iou_predictions, dim=1)  # [B,N,1]

        masks = masks.squeeze(2)   # [B,N,H,W]

        return {
            "boxes": det_out["boxes"],
            "scores": det_out["scores"],
            "labels": det_out["labels"],
            "masks": masks,
            "low_res_masks": low_res_masks,
            "iou_predictions": iou_predictions,
        }