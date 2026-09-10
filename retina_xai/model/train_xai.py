import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class RetiNetraModel(nn.Module):
    """
    EfficientNet-B0 architecture for 5-class Diabetic Retinopathy classification.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(RetiNetraModel, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def load_trained_model(weights_path=None, device="cpu"):
    model = RetiNetraModel(num_classes=5, pretrained=True)
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception:
            pass
    model.to(device)
    model.eval()
    return model