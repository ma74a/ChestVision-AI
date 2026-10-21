import torch
import torch.nn as nn
from torchvision import models


# class DenseNet121(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super().__init__()
#         self.model = models.densenet121(
#             weights=models.DenseNet121_Weights.IMAGENET1K_V1
#         )
#         in_features = self.model.classifier.in_features
#         self.model.classifier = nn.Linear(
#             in_features=in_features,
#             out_features=num_classes
#         )
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)



class DenseNet121(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2, pretrained=True):
        super().__init__()
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=w)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m.classifier.in_features, num_classes),
        )

    def forward(self, x):
        x = torch.relu(self.features(x))   # DenseNet needs this ReLU before pooling
        x = self.pool(x).flatten(1)
        return self.head(x)



if __name__ == "__main__":
    model = DenseNet121(14)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out.shape)
    # print(model.param_groups())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")  # ~7M