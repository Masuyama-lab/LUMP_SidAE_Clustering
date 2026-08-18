from torchvision.datasets import STL10
import torchvision.transforms as transforms
from datasets.seq_tinyimagenet import base_path
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from augmentations import get_aug

from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image


class SequentialSTL10(ContinualDataset):

    NAME = 'seq-stl10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    def get_data_loaders(self, args):
        transform = get_aug(train=True, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

        train_dataset = STL10(base_path() + 'STL10', split='train',
                              download=True, transform=transform)

        memory_dataset = STL10(base_path() + 'STL10', split='train',
                               download=True, transform=test_transform)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
            memory_dataset, _ = get_train_val(memory_dataset, test_transform, self.NAME)
        else:
            test_dataset = STL10(base_path() + 'STL10', split='test',
                                 download=True, transform=test_transform)

        if hasattr(train_dataset, 'labels'):
            train_dataset.targets = torch.tensor(train_dataset.labels, dtype=torch.int64)
        if hasattr(test_dataset, 'labels'):
            test_dataset.targets = torch.tensor(test_dataset.labels, dtype=torch.int64)
        if hasattr(memory_dataset, 'labels'):
            memory_dataset.targets = torch.tensor(memory_dataset.labels, dtype=torch.int64)

        train, memory, test = store_masked_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test

    def get_transform(self, args):
        stl10_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

        def wrap_transform(transform):
            def _transform(img):
                if isinstance(img, torch.Tensor):
                    img = to_pil_image(img)  # Convert Tensor to PIL Image
                return transform(img)

            return _transform

        if args.cl_default:
            original_transform = transforms.Compose([
                # transforms.Resize((32, 32)),  # Resize image to 32x32
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stl10_norm)
            ])
        else:
            original_transform = transforms.Compose([
                # transforms.Resize((32, 32)),  # Resize image to 32x32
                # transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                #                              interpolation=Image.BICUBIC),
                transforms.RandomResizedCrop(96, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                                             interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stl10_norm)
            ])

        return wrap_transform(original_transform)

    def not_aug_dataloader(self, batch_size):
        stl10_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(*stl10_norm)])

        train_dataset = STL10(base_path() + 'STL10', split='train',
                              download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader
