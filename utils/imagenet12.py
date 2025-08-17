
from torchvision import datasets, transforms

class ImageNet12Dataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, train=True):
        if train:
            root = '/home/shunjie/codes/defend_label_inference/cs/Datasets/imagenet12/train'
        else:
            root = '/home/shunjie/codes/defend_label_inference/cs/Datasets/imagenet12/val'
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label
    