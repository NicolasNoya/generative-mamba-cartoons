#%%
from datasets import load_dataset

dset = load_dataset('MultimodalUniverse/jwst', 
                    split='train')

example = next(iter(dset))
#%%
print(example)
#%%
import numpy as np
import matplotlib.pyplot as plt
bands = example["image"]["band"]
flux = example["image"]["flux"]  # shape: (num_bands, height, width)

# simple RGB composite: choose 3 bands
rgb = np.stack([flux[0], flux[3], flux[6]], axis=-1)

# normalize to [0,1] for display
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.imshow(rgb)
plt.show()
#%%
from datasets import load_dataset

# load the dataset in streaming mode
ds_stream = load_dataset(
    "MultimodalUniverse/jwst",
    split="train",
    streaming=True
)

# take only the first 1000 examples
from itertools import islice

first_1000 = list(islice(ds_stream, 1000))
print(f"Number of examples: {len(first_1000)}")

#%%
# example JWST entry
example = first_1000[150]  # streamed entry
print(example)
bands = example["image"]["band"]
flux = example["image"]["flux"]  # shape: (num_bands, height, width)

# simple RGB composite: choose 3 bands
rgb = np.stack([flux[0], flux[3], flux[6]], axis=-1)

# normalize to [0,1] for display
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.imshow(rgb)
plt.show()
# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("alfaro96/los-simpson")

print("Path to dataset files:", path)

# %%
import os
import zipfile

# path from kagglehub.dataset_download
zip_path = path  # e.g., "./los-simpson.zip"
extract_dir = "./los_simpson_images"

# make sure the output directory exists
os.makedirs(extract_dir, exist_ok=True)

# unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Images extracted to:", extract_dir)
#%%
