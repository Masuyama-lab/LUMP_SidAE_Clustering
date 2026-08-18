import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import argparse
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from datasets import get_dataset_ood
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple
from itertools import zip_longest

from utils.buffer import Buffer
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.net.load_state_dict(checkpoint['state_dict'])
    return model

def format_results(results):
    formatted_results = []
    for task_result in results:
        if not isinstance(task_result, list):
            task_result = [task_result]
        formatted_task_result = ["{:.3f}".format(num) for num in task_result]
        formatted_results.append("\t".join(formatted_task_result))
    return "\n".join(formatted_results)


def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if classifier is not None:
                outputs = classifier(outputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def main(device, args):

    results = {'knn-average-acc': [],
               'knn-task-each-acc': [], }

    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)

    args.n_tasks = dataset.N_TASKS  # Number of tasks

    # load OOD dataset
    dataset_ood = get_dataset_ood(args)
    train_loaders_ood, memory_loaders_ood, test_loaders_ood = [], [], []
    for t in range(dataset_ood.N_TASKS):
        tr_ood, me_ood, te_ood = dataset_ood.get_data_loaders(args)
        train_loaders_ood.append(tr_ood)
        memory_loaders_ood.append(me_ood)
        test_loaders_ood.append(te_ood)

    # define model
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))

    # Test OOD dataset
    if args.train.knn_monitor:
        for trained_task in range(dataset.N_TASKS):
            trained_model_path = os.path.join(args.trained_model_dir, f"seed{args.seed}_task{trained_task}.pth")
            model = load_checkpoint(model, trained_model_path)

            knn_acc_list = []
            for i in range(dataset_ood.N_TASKS):
                acc, acc_mask = knn_monitor(model.net.module.backbone, dataset_ood, dataset_ood.memory_loaders[i], dataset_ood.test_loaders[i],
                                            device, args.cl_default, task_id=i, k=min(args.train.knn_k, len(range(dataset_ood.N_TASKS))))
                knn_acc_list.append(acc)

            # preserve results
            results['knn-task-each-acc'].append(knn_acc_list)
            results['knn-average-acc'].append(np.mean(knn_acc_list))

            print('-------results (acc, mean(acc))')
            print(knn_acc_list)
            print(np.mean(knn_acc_list))

            np.set_printoptions(precision=3)
            clustering = args.model.clustering
            ood_data = args.ood_data_name
            if not os.path.isdir(f'{args.log_dir}/ood_{ood_data}'):
                os.mkdir(f'{args.log_dir}/ood_{ood_data}')

            with open(os.path.join(f'{args.log_dir}/ood_{ood_data}', f"knn-task-each-acc.txt"), 'w+') as f:
                f.write(format_results(results['knn-task-each-acc']))
            with open(os.path.join(f'{args.log_dir}/ood_{ood_data}', f"knn-average-acc.txt"), 'w+') as f:
                f.write(format_results(results['knn-average-acc']))




if __name__ == "__main__":

    nTrials = 1  # Number of trials

    for trial in range(nTrials):
        args = get_args(trial=trial)
        main(device=args.device, args=args)
        completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
        os.rename(args.log_dir, completed_log_dir)
        print(f'Log file has been saved to {completed_log_dir}')
