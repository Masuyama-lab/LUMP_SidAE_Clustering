from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
from augmentations import get_aug
from PIL import Image


class SequentialFlower102(ContinualDataset):
    NAME = 'seq-flower'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5

    def get_data_loaders(self, args):
        transform = get_aug(train=True, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

        train_dataset = ImageFolder(base_path() + 'Flower102/train', transform=transform)
        memory_dataset = ImageFolder(base_path() + 'Flower102/train', transform=test_transform)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
            memory_dataset, _ = get_train_val(memory_dataset, test_transform, self.NAME)
        else:
            test_dataset = ImageFolder(base_path() + 'Flower102/test', transform=test_transform)

        train, memory, test = store_masked_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test

    def get_transform(self, args):
        flower_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]  # ImageNetの平均と標準偏差
        if args.cl_default:
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(*flower_norm)])
        else:
            transform = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(*flower_norm)])

        return transform

    def not_aug_dataloader(self, batch_size):
        flower_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]  # ImageNetの平均と標準偏差
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*flower_norm)])

        train_dataset = ImageFolder(base_path() + 'Flower102/train', transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader
