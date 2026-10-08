import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import pandas as pd
from pathlib import Path
from typing import Tuple
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        df: pd.DataFrame,
        transforms: T,
        cfg
    ) -> None:
        self.images_dir = Path(images_dir)
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        
        self.classes = cfg["model"]["classes"]
        self.class_to_idx = {
            clss: idx
            for idx, clss in enumerate(self.classes)
        }
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # return the entire row
        row = self.df.iloc[idx]
        # For Image
        img_name = row["Image Index"]
        img_path = self.images_dir / img_name
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
            
        # For Label
        labels = torch.zeros(len(self.classes), dtype=torch.float32)
        if row["Finding Labels"] != "No Finding":
            # "Effusion"	['Effusion']
            # "Atelectasis|Effusion"	['Atelectasis', 'Effusion']
            # "Mass|Nodule|Effusion"	['Mass', 'Nodule', 'Effusion']
            # "No Finding"	nothing printed (the if is skipped)
            findings = row["Finding Labels"].split('|')
            for find in findings:
                cls_idx = self.class_to_idx[find]
                labels[cls_idx] = 1.0
                
        return img, labels
        