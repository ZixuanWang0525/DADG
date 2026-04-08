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
    def predict(self, image_path: str, text_prompt: str) -> Dict[str, Any]:
        image_pil, image_np = self.load_image(image_path)
        detection_results = self.detect_boxes_batch(image_pil, text_prompt)
        masks = self.segment_from_boxes(image_np, detection_results["boxes"])

        return {
            "boxes": detection_results["boxes"],
            "scores": detection_results["scores"],
            "labels": detection_results["labels"],
            "masks": masks,   # [N, H, W]
        }
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path = "./4495.jpg"   
    text_prompt = "a sofa. a television"         
    grounding_model_dir = "../checkpoints/grounding_dino"
    sam_checkpoint_path = "../checkpoints/sam/sam_vit_b_01ec64.pth"

    model = GroundingSAM(
        grounding_model_dir=grounding_model_dir,
        sam_checkpoint_path=sam_checkpoint_path,
        sam_type="vit_b",
        device=device,
        box_threshold=0.35,
        text_threshold=0.25,
    )

    result = model.predict(image_path, text_prompt)

    print("labels:", result["labels"])
    print("boxes :", result["boxes"])
    print("scores:", result["scores"])
    print("masks shape:", result["masks"].shape)