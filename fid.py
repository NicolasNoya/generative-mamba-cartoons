#%%
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
CKPT_PATH = "checkpoints/simpsons-lora"  # trained checkpoint
DEVICE = "cuda"
# ──────────────────────────────────────────────────────────────────────────
def detect_target_modules(state_dict: dict) -> list:
    """
    Infer which modules were LoRA-wrapped from the checkpoint keys.
    A LoRA-wrapped module has keys ending in '.lora_A.default.weight'.
    e.g. 'model.base_model.model.mamba.lm_head.lora_A.default.weight'
         → module leaf name is 'lm_head'
    """
    target_modules = set()
    for key in state_dict:
        if "lora_A.default.weight" in key:
            # key looks like: ....<module_name>.lora_A.default.weight
            module_name = key.split(".lora_A.default.weight")[0].split(".")[-1]
            target_modules.add(module_name)
    modules = sorted(target_modules)
    print(f"  detected LoRA target_modules from checkpoint: {modules}")
    return modules

def load_model(ckpt_path: str, device: str) -> MambaWrapper:
    weights_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"pytorch_model.bin not found in {ckpt_path}.\n"
            "Make sure save_safetensors=False is set in TrainingArguments."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    # Reconstruct with the exact same LoRA target_modules used during training
    target_modules = detect_target_modules(state_dict)
    model = MambaWrapper(target_modules=target_modules)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(
        f"  trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    return model

model = load_model(CKPT_PATH, "cuda")

os.makedirs(GEN_IMG_DIR, exist_ok=True)

# Load model
# model = MambaWrapper(
#     target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"]
# )
# model.load_state_dict(torch.load(CKPT_PATH))
# model = model.to(DEVICE).eval()

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

# %%
