'''
dataloader.py

This file contains the function that allows you to load skin cancer data from folders.

'''

from torch.utils.data import Dataset
from torchvision import transforms
import os
import glob
import random
from PIL import Image

class SkinCancerDataloader(Dataset):
    def __init__(self, data_dir, transform=None, seed=0, resize=(128, 128)):
        """
        Skin Cancer Dataloader
        
        Args:
            data_dir: Directory containing 'benign' and 'malignant' folders
            transform: Optional transforms
            seed: Random seed for reproducibility
            resize: Image resize dimensions
        """
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Load benign images (label 0)
        benign_dir = os.path.join(data_dir, 'benign')
        if os.path.exists(benign_dir):
            benign_images = glob.glob(os.path.join(benign_dir, '*.jpg'))
            self.image_paths.extend(benign_images)
            self.labels.extend([0] * len(benign_images))
        
        # Load malignant images (label 1)
        malignant_dir = os.path.join(data_dir, 'malignant')
        if os.path.exists(malignant_dir):
            malignant_images = glob.glob(os.path.join(malignant_dir, '*.jpg'))
            self.image_paths.extend(malignant_images)
            self.labels.extend([1] * len(malignant_images))
        
        # Shuffle data
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        
        # Convert back to lists
        self.image_paths = list(self.image_paths)
        self.labels = list(self.labels)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label