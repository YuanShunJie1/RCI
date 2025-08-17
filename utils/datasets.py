from torchvision import datasets, transforms
# from torch.utils.data import Dataset
# import os
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# import torchvision
# from PIL import Image
# from torchvision.datasets import VisionDataset
# from sklearn.model_selection import train_test_split
# import numpy as np
import utils.imagenet12 as imagenet12
import utils.letter as letter
import utils.yeast as yeast

datasets_choices = [
    "mnist",
    "fashionmnist",
    "cifar10",
    "cifar100",
    "yeast",
    "letter",
    "imagenet12"
]

datasets_name = {
    "mnist": "mnist",
    "fashionmnist": "fashionmnist",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet12": "imagenet12",
    "yeast": "yeast",
    "letter": "letter"
}

from torchvision.datasets import CIFAR10, CIFAR100
datasets_dict = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    # "cifar10": datasets.CIFAR10,
    "cifar10": CIFAR10,
    # "cifar100": datasets.CIFAR100,
    "cifar100": CIFAR100,
    
    'imagenet12': imagenet12.ImageNet12Dataset,
    'yeast': yeast.YeastDataset,
    'letter': letter.LetterDataset,
}

datasets_classes = {
    "mnist": 10,
    "fashionmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    'imagenet12': 12,
    'yeast': 10,
    'letter': 26,
}

normalize_imagenet12 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize_cifar = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])

transforms_default = {
    "mnist": transforms.Compose([transforms.ToTensor()]),
    "fashionmnist": transforms.Compose([transforms.ToTensor()]),
    "cifar10": transforms.Compose([transforms.ToTensor(), normalize_cifar]),
    "cifar100": transforms.Compose([transforms.ToTensor(), normalize_cifar]),
    'yeast': None,
    'letter': None,
    'imagenet12':transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize_imagenet12]),
}
