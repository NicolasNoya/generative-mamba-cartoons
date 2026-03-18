#%%
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import glob

class SimpsonsDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256)):
        """
        Args:
            root_dir (str): Path to 'train' or 'test' folder
        """
        self.root_dir = root_dir
        self.target_size = target_size
        # Get all image files recursively
        self.image_paths = glob.glob(os.path.join(root_dir, "**/*.jpg"), recursive=True)
        self.image_paths += glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)

        # Map folder names to integer labels
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Simple transform: convert to tensor and scale to [0,1]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)

        image = F.interpolate(
            image,
            size=self.target_size,
            mode='bilinear',      # good for images
            align_corners=False
        )[0]

        # Get label from parent folder name
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[label_name]

        return image, label

# Example usage
train_dataset = SimpsonsDataset("mamba_generative/data/train")
test_dataset = SimpsonsDataset("mamba_generative/data/test")

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of test images: {len(test_dataset)}")
#%%
import numpy as np
import matplotlib.pyplot as plt
img = train_dataset[30][0]
# Convert to NumPy array
img = img.numpy()

# Transpose to (H, W, C)
img = np.transpose(img, (1, 2, 0))
plt.imshow(img)
plt.axis("off")
plt.show()

#%%
# Example DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)
print("Labels:", labels)