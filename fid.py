import sys, torch

sys.path.append(".")  # run from repo root

from cleanfid import fid
from models.mambawrapper import MambaWrapper
from torchvision.utils import save_image
import os

# ── config ────────────────────────────────────────────────────────────────
REAL_IMG_DIR = "data/test"  # folder of real Simpsons images
GEN_IMG_DIR = "research/fid_gen"  # where generated images will be saved
N_GENERATE = 2048  # match or exceed the test set size
BATCH_SIZE = 8
CKPT_PATH = "checkpoints/simpsons-lora/..."  # trained checkpoint
DEVICE = "cuda"
# ──────────────────────────────────────────────────────────────────────────

os.makedirs(GEN_IMG_DIR, exist_ok=True)

# Load model
model = MambaWrapper(
    target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"]
)
model.load_state_dict(torch.load(CKPT_PATH))
model = model.to(DEVICE).eval()

# Generate images
print(f"Generating {N_GENERATE} images...")
img_idx = 0
with torch.no_grad():
    for _ in range(N_GENERATE // BATCH_SIZE):
        imgs = model.generate(
            batch=BATCH_SIZE,
            temperature=1.0,
            top_k=600,
            top_p=0.98,
            cfg_scale=1.0,
        )
        # imgs in [-1, 1] → [0, 1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        for img in imgs:
            save_image(img, os.path.join(GEN_IMG_DIR, f"{img_idx:05d}.png"))
            img_idx += 1

print(f"Saved {img_idx} images to {GEN_IMG_DIR}")

# Compute FID
score = fid.compute_fid(REAL_IMG_DIR, GEN_IMG_DIR, device=DEVICE, num_workers=4)
print(f"\nFID: {score:.2f}")
