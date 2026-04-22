import os
import urllib.request

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
SAVE_DIR = "../checkpoints/dino"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "dinov2_vits14_pretrain.pth")

urllib.request.urlretrieve(URL, SAVE_PATH)
print(f"saved to: {SAVE_PATH}")