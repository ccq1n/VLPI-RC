import torch
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def ResNet18(num_classes, weights=None):
    return resnet18(num_classes=num_classes, weights=weights)

def ResNet34(num_classes, weights=None):
    return resnet34(num_classes=num_classes, weights=weights)

if __name__ == '__main__':
    model = ResNet34(num_classes=2)
    x = torch.randn((10, 3, 224, 224))
    y = model(x)
    print(y.size())