import timm
import torch
import torch.nn as nn

class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int,
                 name="vit_small_patch16_224.augreg_in21k_ft_in1k",
                 drop_path=0.1, dropout=0.1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0,   # num_classes=0 -> returns pooled features
            drop_path_rate=drop_path,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))
    
    
if __name__ == "__main__":
    model = ViTClassifier(14)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out.shape)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")  # ~7M