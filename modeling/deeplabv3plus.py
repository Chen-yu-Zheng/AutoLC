from segmentation_models_pytorch import DeepLabV3Plus
import torch

def Deeplabv3plus(encoder_name, encoder_weights, classes):
    return DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=classes)

if __name__ == '__main__':
    net = Deeplabv3plus(encoder_name='resnet50', encoder_weights=None, classes=7)
    img = torch.rand((2,3,768,768))
    print(net(img).shape)
    