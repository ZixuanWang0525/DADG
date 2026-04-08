import os
import urllib.request

SAVE_DIR = "../checkpoints/sam"
os.makedirs(SAVE_DIR, exist_ok=True)

URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAVE_PATH = os.path.join(SAVE_DIR, "sam_vit_b_01ec64.pth")

urllib.request.urlretrieve(URL, SAVE_PATH)
print(f"Saved to: {SAVE_PATH}")