#%%
import sys
import torch

sys.path.append("/home/onyxia/work/AiM")

from models.aim import AiM

model = AiM.from_pretrained("hp-l33/aim-xlarge")
model = model.cuda().half()
model.eval()
#%%
c = torch.tensor([281], device=model.mamba.lm_head.weight.device)
with torch.no_grad():
    imgs = model.generate(
        batch=1,      # T4 safe
        temperature=0,
        top_p=0.98,
        top_k=600,
        cfg_scale=1,
        c=c
    )

import matplotlib.pyplot as plt

img = imgs[0].detach().cpu()

# If in [-1,1], normalize
if img.min() < 0:
    img = (img + 1) / 2

# Convert to float32
img = img.float()

# CHW → HWC
img = img.permute(1, 2, 0)

plt.imshow(img)
plt.axis("off")
plt.show()