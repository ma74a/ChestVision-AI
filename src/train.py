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
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

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
    
    
def saving_training_plots(history_df, lr, output_dir):
    """
    Function to store train vs. val loss and train vs. val accuracy.

    Input : history df with columns 
                'train loss','train acc','val loss','val acc','learning rate'
    Output : NONe
        
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].plot(history_df["train loss"], label="Train Loss")
    ax[0].plot(history_df["val loss"], label="Validation Loss")
    ax[0].set_title("Loss Curves")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history_df["train acc"], label="Train Accuracy")
    ax[1].plot(history_df["val acc"], label="Validation Accuracy")
    ax[1].set_title("Accuracy Curves")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    ax[2].plot(lr, label="Learning rate per step")
    ax[2].set_title("Learning Rate Curve")
    ax[2].set_xlabel("steps")
    ax[2].legend()

    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path)
    log.info(f"Saved training plot to {plot_path}")
    
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
    dense_model = DenseNet121(num_classes=cfg["model"]["num_classes"]).to(device)
    test_input = torch.randn(
        cfg["train"]["batch_size"],
        cfg["model"]["in_channels"],
        cfg["model"]["img_size"],
        cfg["model"]["img_size"]
    )
    
    log.info("--- Model Summary ---")
    log.info(summary(dense_model, input_data=test_input))
    log.info("--------------------")
    
    # get accuracy
    def accuracy_fn(y_pred, y_true):
        # y_pred: [B, 14] — raw logits
        # y_true: [B, 14] — binary labels
        probs = torch.sigmoid(y_pred)
        preds = (probs >= 0.7).float()
        
        correct = (preds == y_true).sum().item()
        # numel returns the total nums in y_true
        total = y_true.numel()

        return correct / total

    
    optimizer = torch.optim.AdamW(
        dense_model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
        eta_min=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss()
    
    log.info("Model training...")
    train_losses = []
    train_acces = []
    val_losses = []
    val_acces = []
    learning_rates = []
    n_trains = len(train_loader)
    n_vals = len(val_loader)
    best_val_acc = 0
    for epoch in range(cfg["train"]["epochs"]):
        train_loss_avg = 0
        train_acc_avg = 0
        dense_model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            learning_rates.append(scheduler.get_last_lr()[0])
            
            train_logits = dense_model(images)
            # print(f"train_logits shape -> : {train_logits.shape}")
            # print(f"labels shape -> : {labels.shape}")
            loss = criterion(train_logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss_avg += loss.item()
            train_acc_avg += accuracy_fn(train_logits, labels)
            
        epoch_train_avg_loss = train_loss_avg / n_trains
        epoch_train_avg_acc = train_acc_avg / n_trains
        train_losses.append(epoch_train_avg_loss)
        train_acces.append(epoch_train_avg_acc)
        
        val_loss_avg = 0
        val_acc_avg = 0
        dense_model.eval()
        with torch.inference_mode():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                
                val_logits = dense_model(images)
                loss = criterion(val_logits, labels)
                
                val_loss_avg += loss.item()
                val_acc_avg += accuracy_fn(val_logits, labels)
                
            epoch_val_avg_loss = val_loss_avg / n_vals
            epoch_val_avg_acc = val_acc_avg / n_vals
            val_losses.append(epoch_val_avg_loss)
            val_acces.append(epoch_val_avg_acc)
            
            print(
                f"Epoch {epoch+1} | "
                f"train_loss: {epoch_train_avg_loss:.4f} | train_accuracy: {epoch_train_avg_acc:.4f} | "
                f"val_loss: {epoch_val_avg_loss:.4f} | val_accuracy: {epoch_val_avg_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            
            log.info(f"Epoch {epoch+1} | "
                f"train_loss: {epoch_train_avg_loss:.4f} | train_accuracy: {epoch_train_avg_acc:.4f} | "
                f"val_loss: {epoch_val_avg_loss:.4f} | val_accuracy: {epoch_val_avg_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            
            # storing the model with best val
            if best_val_acc > epoch_val_avg_loss:
                best_val_acc = epoch_val_avg_loss
                model_path = os.path.join(output_dir, "best_dense_model.pt")
                torch.save(dense_model.state_dict(), model_path)
                log.info(f"New best model saved to {model_path} , with accuracy : {best_val_acc}")
                
                
    learning_rates.append(scheduler.get_last_lr()[0])

    log.info("Storing the training artifacts detials")

    data = list(zip(train_losses,train_acces,val_losses,val_acces))
    df = pd.DataFrame(data,columns=['train loss','train acc','val loss','val acc'])
    
    csv_path = os.path.join(output_dir, "training_history.csv")
    df.to_csv(csv_path,index_label="epoch")
    log.info("WOrking on Training Plots...")
    saving_training_plots(df,learning_rates,output_dir)
    df = {'learning_rates_per_step':learning_rates}
    df = pd.DataFrame(df)
    csv_path = os.path.join(output_dir, "learning_rates_per_step.csv")
    df.to_csv(csv_path,index_label="steps")

    log.info(f"😎 Training Completed, details stored in {output_dir}")
        
    
    
if __name__ == "__main__":
    main_train()