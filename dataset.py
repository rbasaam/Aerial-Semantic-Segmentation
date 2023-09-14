import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class aerialSemantics(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images) 
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("image", "graymask").replace(".jpg",".png"))
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L")  , dtype=np.uint8)
        return image, mask

class BatchPatchLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_batches, patch_size, num_workers, pin_memory):
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.num_batches = num_batches
        self.patch_size = patch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self._get_batch()

    def _get_batch(self):
        data_batch = []
        targets_batch = []

        for _ in range(self.batch_size):
            image, mask = self.dataset[np.random.randint(len(self.dataset))]

            h, w, k = image.shape

            if h < self.patch_size or w < self.patch_size:
                print("Patch Size Too Big")
                continue

            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)

            image_patch = np.transpose(image[top:top + self.patch_size, left:left + self.patch_size, :],(2, 0, 1)).astype(np.float32)
            mask_patch = mask[top:top + self.patch_size, left:left + self.patch_size]

            if not np.any(image_patch) or not np.any(mask_patch):
                continue
            
            data_batch.append(torch.from_numpy(image_patch))
            targets_batch.append(torch.from_numpy(mask_patch))

            return torch.stack(data_batch), torch.stack(targets_batch)