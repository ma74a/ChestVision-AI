import torch
import torch.nn as nn
from torch import optim
from torchinfo import summary

import logging
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import json

from get_data import split_load_data
from cnn import DenseNet121


log = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
    device = torch.device("cpu")
    
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main_train(cfg: DictConfig):
    # print and log the active config and the Hydra output directory:
    print(f"Current working directory: {os.getcwd()}")
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"All artifacts will be saved in {output_dir}")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    log.info("Dataset creation begin")
    train_dataset, val_dataset, train_loader, val_loader = split_load_data(cfg=cfg)
    num_classes = len(train_dataset.classes)
    log.info("Dataset created")
    
    log.info(f"Dataset Classes and Corresponding Labels : {train_dataset.class_to_idx}")
    
    try:
        log.info("Verifying consistency between config and dataset...")
        assert num_classes == cfg["model"]["num_classes"], \
            f"Mismatch: config expects {cfg.model.num_classes} classes, but dataset has {num_classes}."
        log.info("✅ Verification successful.")

    except AssertionError as e:
        log.error(f"CONFIGURATION ERROR: {e}")
        import sys
        sys.exit(1)
        
    log.info("Storing the index vs label mapping for the Current Dataset")
    idx_to_cls = {clss: idx for clss, idx in train_dataset.class_to_idx.items()}
    # print(idx_to_cls)
    mapping_save_path = os.path.join(output_dir, "mapping_saved_file.json")
    with open(mapping_save_path, 'w+') as f:
        json.dump(idx_to_cls, f, indent=4)
    log.info(f"Mapping saved at : {mapping_save_path}")
    
    log.info("Model Creation Begin")
    dense_model = DenseNet121(num_classes=cfg["model"]["num_classes"])
    test_input = torch.randn(
        cfg["train"]["batch_size"],
        cfg["model"]["in_channels"],
        cfg["model"]["img_size"],
        cfg["model"]["img_size"]
    )
    
    log.info("--- Model Summary ---")
    log.info(summary(dense_model, input_data=test_input))
    log.info("--------------------")
    
    
    
if __name__ == "__main__":
    main_train()