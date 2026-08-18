import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SemiconDataset(Dataset):
    """
    PyTorch Dataset for Semiconductor Inspection Image Restoration & 2x Super-Resolution.
    
    Loads paired .npy files from NoisyLR and GT folders.
    - NoisyLR Input:  (1, 128, 128) float32 clipped to [0.0, 1.0]
    - GT Target:      (1, 256, 256) float32 in [0.0, 1.0]
    """
    def __init__(self, data_root, split='train', train_ratio=0.9, seed=42, augment=False):
        super(SemiconDataset, self).__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.augment = augment
        
        # Path resolution
        train_base = self.data_root / "train" / "train"
        if not train_base.exists():
            train_base = self.data_root / "train"
            
        self.gt_dir = train_base / "GT"
        self.noisy_dir = train_base / "NoisyLR"
        
        if not self.gt_dir.exists() or not self.noisy_dir.exists():
            raise FileNotFoundError(f"Dataset directories not found under {self.data_root}. Expected GT and NoisyLR subfolders.")
            
        # Match filenames strictly
        gt_files = set(p.name for p in self.gt_dir.glob("*.npy"))
        noisy_files = set(p.name for p in self.noisy_dir.glob("*.npy"))
        
        common_filenames = sorted(list(gt_files.intersection(noisy_files)))
        if not common_filenames:
            raise RuntimeError(f"No matching .npy paired files found between {self.gt_dir} and {self.noisy_dir}")
            
        # Reproducible Train/Val Split (90% train = 2880 pairs, 10% val = 320 pairs)
        rng = random.Random(seed)
        shuffled_filenames = common_filenames.copy()
        rng.shuffle(shuffled_filenames)
        
        num_total = len(shuffled_filenames)
        num_train = int(num_total * train_ratio)
        
        if split == 'train':
            self.filenames = shuffled_filenames[:num_train]
        elif split == 'val' or split == 'validation':
            self.filenames = shuffled_filenames[num_train:]
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'val'.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        gt_path = self.gt_dir / fname
        noisy_path = self.noisy_dir / fname
        
        # Load .npy arrays using numpy.load()
        gt_arr = np.load(gt_path).astype(np.float32)       # Shape: (256, 256)
        noisy_arr = np.load(noisy_path).astype(np.float32) # Shape: (128, 128)
        
        # Clip ONLY input (NoisyLR) to [0.0, 1.0] before feeding to model
        noisy_arr = np.clip(noisy_arr, 0.0, 1.0)
        
        # Synchronized Geometric Data Augmentations (Train mode only)
        if self.augment and self.split == 'train':
            # 1. Random Horizontal Flip
            if random.random() > 0.5:
                noisy_arr = np.flip(noisy_arr, axis=1)
                gt_arr = np.flip(gt_arr, axis=1)
                
            # 2. Random Vertical Flip
            if random.random() > 0.5:
                noisy_arr = np.flip(noisy_arr, axis=0)
                gt_arr = np.flip(gt_arr, axis=0)
                
            # 3. Random 90-Degree Rotation
            k = random.randint(0, 3)
            if k > 0:
                noisy_arr = np.rot90(noisy_arr, k)
                gt_arr = np.rot90(gt_arr, k)
                
            noisy_arr = noisy_arr.copy()
            gt_arr = gt_arr.copy()
            
        # Add channel dimension: Input (1, 128, 128), Target (1, 256, 256)
        noisy_tensor = torch.from_numpy(np.expand_dims(noisy_arr, axis=0)).float()
        gt_tensor = torch.from_numpy(np.expand_dims(gt_arr, axis=0)).float()
        
        return noisy_tensor, gt_tensor, fname


def get_dataloaders(data_root, batch_size=8, seed=42, num_workers=0, train_augment=True):
    """
    Utility function to create train and validation DataLoaders.
    """
    train_dataset = SemiconDataset(data_root=data_root, split='train', seed=seed, augment=train_augment)
    val_dataset = SemiconDataset(data_root=data_root, split='val', seed=seed, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader
