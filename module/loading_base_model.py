from utils.open_vocabulary_detection import GradientDetection
from transformers import CLIPTokenizer, CLIPModel

class Base_models:
    def __init__(
            self,
            grounding_model_dir=None,
            sam_checkpoint_path=None,
            CLIP_dir=None,
            device="cuda",
            image_size: int = 512,
            query_temperature: float = 0.07,
            sam_model_type: str = "vit_b",
    ):
        self.device = device
        if grounding_model_dir is not None and sam_checkpoint_path is not None:
                self.gradient_detector = GradientDetection(
                grounding_model_dir=grounding_model_dir,
                sam_checkpoint_path=sam_checkpoint_path,
                device=device,
                image_size=image_size,
                query_temperature=query_temperature,
                sam_model_type=sam_model_type,
            )
        if CLIP_dir is not None:
             self.CLIPModel = CLIPModel.from_pretrained(CLIP_dir)
             self.CLIPTokenizer = CLIPTokenizer.from_pretrained(CLIP_dir)
