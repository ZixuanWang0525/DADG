import os
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-base"
SAVE_DIR = "../checkpoints/grounding_dino"

os.makedirs(SAVE_DIR, exist_ok=True)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=SAVE_DIR
)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID,
    cache_dir=SAVE_DIR
)

processor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"Grounding DINO saved to: {SAVE_DIR}")