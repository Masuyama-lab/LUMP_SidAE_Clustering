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
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from pyclustering.nnet.som import som, type_conn, type_init, som_parameters
from utils.ca_plus import ClusterCAplus

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
    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
    results = {'knn-cls-acc': [],
               'knn-cls-each-acc': [],
               'knn-cls-max-acc': [],
               'knn-cls-fgt': [],}

    args.n_tasks = dataset.N_TASKS  # Number of tasks

    # define model
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))

    logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)

    # setup replay buffer
    buffer = Buffer(args.model.buffer_size, device)
    clustering = args.model.clustering

    train_loaders, memory_loaders, test_loaders = [], [], []
    for t in range(dataset.N_TASKS):
        tr, me, te = dataset.get_data_loaders(args)
        train_loaders.append(tr)
        memory_loaders.append(me)
        test_loaders.append(te)

    for t in range(dataset.N_TASKS):
        # print(args.eval.type)
        # train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
        # NOTE: set eval_type to 'all' to evaluate on all tasks and get the average forgetting
        if args.eval.type == 'all':
            eval_tids = [j for j in range(dataset.N_TASKS)]
        elif args.eval.type == 'curr':
            eval_tids = [t]
        elif args.eval.type == 'accum':
            eval_tids = [j for j in range(t + 1)]
        else:
            sys.exit('Stopped!! Wrong eval-type.')

        if clustering == "kmeans":
            clustering_model = KMeans(n_clusters=args.train.batch_size, n_init=10, init="k-means++")
        elif clustering == "kmedoids":
            clustering_model = KMedoids(n_clusters=args.train.batch_size, init="k-medoids++")
        elif clustering == "sc":
            clustering_model = SpectralClustering(n_clusters=args.train.batch_size, n_init=10,
                                                  affinity="nearest_neighbors")
        elif clustering == "gmm":
            clustering_model = GaussianMixture(n_components=args.train.batch_size, n_init=10, init_params="k-means++")
        elif clustering == "som":
            parameters = som_parameters()
            rows = 16
            cols = 16
            structure = type_conn.grid_four
            clustering_model = som(rows, cols, structure, parameters)
        elif clustering == "caplus":
            clustering_model = ClusterCAplus(max_nodes_=args.model.buffer_size)
        else:
            clustering_model = KMeans(n_clusters=args.train.batch_size, n_init=10, init="k-means++")

        latent_variables_list = []
        notaug_images_list = []
        images1_list = []
        images2_list = []

        global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
        for epoch in global_progress:
            model.train()

            local_progress = tqdm(train_loaders[t], desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
            for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
                data_dict, buffer = model.observe(images1, labels, images2, notaug_images, buffer, epoch)
                logger.update_scalers(data_dict)


            if (epoch + 1 == args.train.stop_at_epoch):
                # for kmeans
                model.eval()
                with torch.no_grad():
                    for _, ((images1, images2, notaug_images), labels) in enumerate(train_loaders[t]):

                        # Extract latent variables on the t-th task dataset by an encoder
                        encoder = model.net.module.encoder
                        latent_variables_batch = encoder(notaug_images.to(device))
                        latent_variables_list.append(latent_variables_batch.detach().cpu())

                        notaug_images_list.append(notaug_images)
                        images1_list.append(images1)
                        images2_list.append(images2)

                # run clustering on the t-th task dataset
                latent_variables_cat = torch.cat(latent_variables_list, dim=0)
                # kmeans.fit(latent_variables_cat.cpu().numpy())
                clustering_model.fit(latent_variables_cat.cpu().numpy())

                # Retrieve the cluster centers from kmeans or caplus
                if clustering == "kmeans" or clustering == "kmedoids":
                    centroids = clustering_model.cluster_centers_
                elif clustering == "gmm":
                    centroids = clustering_model.means_
                    del clustering_model
                    # clustering_model = GaussianMixture(n_components=args.train.batch_size, n_init=10, init_params="k-means++")
                elif clustering == "sc":
                    centroids = []
                    cluster_labels = clustering_model.labels_
                    sample = latent_variables_cat.cpu().numpy()

                    for i in range(args.train.batch_size):
                        cluster_data = np.array([sample[j] for j in range(len(sample)) if i == cluster_labels[j]])

                        centroid = np.mean(cluster_data, axis=0)
                        centroids.append(centroid)

                    centroids = np.array(centroids)
                elif clustering == "som":
                    centroids = np.array(clustering_model.weights)
                elif clustering == "caplus":
                    centroids = list(
                        clustering_model._CAplus__get_node_attributes_from('weight', list(clustering_model.G_.nodes)))

                # Find the indices of the closest latent variables to each node
                # Use scipy's cdist to calculate the distance between each latent_variable and each node
                buffer_node = torch.from_numpy(np.array(centroids))

                distances = cdist(centroids, latent_variables_cat.cpu().numpy(), metric='euclidean')
                closest_indices = np.argmin(distances, axis=1)

                # Retrieve the corresponding data from notaug_images_list and images2_list
                notaug_images_cat = torch.cat(notaug_images_list, dim=0)
                images1_cat = torch.cat(images1_list, dim=0)
                images2_cat = torch.cat(images2_list, dim=0)
                closest_notaug_images = notaug_images_cat[closest_indices]
                closest_images1 = images1_cat[closest_indices]
                closest_images2 = images2_cat[closest_indices]

                buf_inputs1 = closest_notaug_images
                buf_inputs2 = closest_images2

                buffer.add_data(examples=buf_inputs1, logits=buf_inputs2)  # for random sampling

            global_progress.set_postfix(data_dict)

            # if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            if (epoch + 1) == args.train.stop_at_epoch:
                # depend on args.eval.type
                if args.train.knn_monitor:
                    knn_acc_list = []
                    for i in eval_tids:
                        acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i],
                                                    device, args.cl_default, task_id=i, k=min(args.train.knn_k, len(eval_tids)))
                        knn_acc_list.append(acc)

                    # preserve results
                    kfgt = []
                    results['knn-cls-each-acc'].append(knn_acc_list)
                    results['knn-cls-max-acc'] = [max(item) for item in zip_longest(*results['knn-cls-each-acc'], fillvalue=0)][:t]
                    for j in range(t):
                        kfgt.append(results['knn-cls-max-acc'][j] - knn_acc_list[j])
                    results['knn-cls-acc'].append(np.mean(knn_acc_list))
                    if len(kfgt) > 0:
                        results['knn-cls-fgt'].append(np.mean(kfgt))

                    print(f'------- task{t} results')
                    print(knn_acc_list)
                    print(f'accuracy: {np.mean(knn_acc_list):.3f}')
                    if len(kfgt) > 0:
                        print(f'forgetting: {np.mean(kfgt):.3f}')


        model_path = os.path.join(args.ckpt_dir, f"seed{args.seed}_task{t}.pth")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.net.state_dict()
        }, model_path)
        print(f"Task Model saved to {model_path}")
        with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
            f.write(f'{model_path}')
        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        np.set_printoptions(precision=3)
        with open(os.path.join(f'{args.log_dir}', f"knn-cls-each-acc.txt"), 'w+') as f:
            f.write(format_results(results['knn-cls-each-acc']))
        with open(os.path.join(f'{args.log_dir}', f"knn-cls-max-acc.txt"), 'w+') as f:
            f.write(format_results(results['knn-cls-max-acc']))
        with open(os.path.join(f'{args.log_dir}', f"knn-cls-acc.txt"), 'w+') as f:
            f.write(format_results(results['knn-cls-acc']))
        with open(os.path.join(f'{args.log_dir}', f"knn-cls-fgt.txt"), 'w+') as f:
            f.write(format_results(results['knn-cls-fgt']))

    if args.eval is not False and args.cl_default is False:
        args.eval_from = model_path


if __name__ == "__main__":
    nTrials = 3  # Number of trials

    for trial in range(nTrials):
        args = get_args(trial=trial)
        main(device=args.device, args=args)
        completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
        os.rename(args.log_dir, completed_log_dir)
        print(f'Log file has been saved to {completed_log_dir}')
