from torchvision.models.segmentation import deeplabv3_resnet50

def Deeplabv3(classes):
    return deeplabv3_resnet50(num_classes=classes)