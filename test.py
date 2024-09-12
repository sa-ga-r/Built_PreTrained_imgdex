import torch
import torchvision.models as models
from torchvision.models import ResNet101_Weights

model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

weights = ResNet101_Weights.IMAGENET1K_V2
#class_name = getattr(model, "class_to_idx_").keys()
class_names = weights.meta['categories']
print(class_names)