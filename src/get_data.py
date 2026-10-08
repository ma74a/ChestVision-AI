from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

from dataset import ChestXRayDataset

from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def split_load_data(cfg):
    # we will split according to patient id
    # not using random_split
    # becuase there are more that one patient has more than one image
    # and if we use random_split there will be data leakage
    
    # read the csv file
    df = pd.read_csv(cfg["train"]["csv_file"])
    
    # get the images in train_val file
    with open(cfg["train"]["train_val_file"], 'r') as f:
        filenames = [
            line.strip()
            for line in f
        ]
        
    # filter the df which we keep only the rows with only matching filenames
    df = df[df["Image Index"].isin(filenames)].copy()
    
    # get the patient id
    patient_ids = df["Patient ID"].unique()
    print("Number of patients:", len(patient_ids))
    # split according patient id
    # [1, 2, 3, 4, 5]   <- each patient appears once, even if they have 10 images
    # after splitting we'll have
    # train -> [3, 1, 5, 2], for val -> [4]
    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=0.2,
        random_state=21,
    )
    df_train = df[df["Patient ID"].isin(train_patients)].copy()
    df_val = df[df["Patient ID"].isin(val_patients)].copy()
    
    # print(len(df_train), len(df_val))
    # check if there is any leakage or not
    overlap = set(df_train["Patient ID"]) & set(df_val["Patient ID"])
    print("Overlapping patients:", len(overlap)) # should be 0
    
    train_dataset = ChestXRayDataset(
        images_dir=cfg["train"]["images_dir"],
        df=df_train,
        transforms=train_transform
    )
    val_dataset = ChestXRayDataset(
        images_dir=cfg["train"]["images_dir"],
        df=df_val,
        transforms=val_transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False
    )
    
    return train_loader, val_loader
    
if __name__ == "__main__":
    with open("/home/etman/etman/ChestVision-AI/configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    split_load_data(cfg=config)