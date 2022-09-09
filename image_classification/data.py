from os.path import join
import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset, TensorDataset, DataLoader
import torchvision.transforms as transforms

from utils import shuffle_tensor


cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifar10_mean = (0.4914, 0.4822, 0.4465) 
cifar10_std = (0.2471, 0.2435, 0.2616)

cinic_mean_RGB = (0.47889522, 0.47227842, 0.43047404)
cinic_std_RGB = (0.24205776, 0.23828046, 0.25874835)

data_transform = lambda mean, std: transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = lambda mean, std: transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def load_cinic10(root):
    print(f'loading data from {root}')
    images = torch.load(join(root, 'images.pt'))
    labels = torch.load(join(root, 'labels.pt'))
    idxs = torch.arange(images.size(0))
    all_datasets = torch.utils.data.TensorDataset(images, labels, idxs)
    return all_datasets

def load_cifar5m(root, num_parts):
    print(f'loading data from {root}')
    images, labels = [], []
    for i in range(num_parts):
        data = np.load(join(root, f'part{i}.npz'))
        images.append(data['X'])
        labels.append(data['Y'])
    images = np.concatenate(images)
    labels = np.concatenate(labels)

    images = torch.tensor(images).permute(0, 3, 1, 2)
    labels = torch.tensor(labels).long()
    idxs = torch.arange(images.size(0))
    all_datasets = torch.utils.data.TensorDataset(images, labels, idxs)
    return all_datasets

def load_data(dataset, data_dir):
    if dataset == 'cifar5m':
        return load_cifar5m(join(data_dir, dataset), num_parts=3)
    elif dataset == 'cinic10':
        return load_cinic10(join(data_dir, dataset))


def get_transforms(dataset):
    transform_train = lambda img_mean, img_std: transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(img_mean, img_std),
    ])
    transform_test = lambda img_mean, img_std: transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(img_mean, img_std),
    ])
    if dataset == 'cifar5m':
        return transform_train(cifar10_mean, cifar10_std), transform_test(cifar10_mean, cifar10_std)
    elif dataset == 'cinic10':
        return transform_train(cinic_mean_RGB, cinic_std_RGB), transform_test(cinic_mean_RGB, cinic_std_RGB)

class CustomDatasetWithTransform:
    def __init__(self, dataset: TensorDataset, transform, transform_index=0):
        self.dataset = dataset
        self.transform = transform
        self.transform_index = transform_index

    def __getitem__(self, index):
        sample = tuple(self.dataset[index])
        image = self.transform(sample[self.transform_index])
        sample = sample[:self.transform_index] + tuple([image]) + sample[self.transform_index+1:]
        return sample

    def __len__(self):
        return len(self.dataset)

class Batches():
    def __init__(self, dataset, transform, batch_size, shuffle, num_workers, drop_last=False):
        self.dataloader = DataLoader(CustomDatasetWithTransform(dataset, transform), batch_size=batch_size, 
                                     num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)
    
    def __iter__(self):
        for x, y, idx in self.dataloader:
            yield {'input': x.cuda().half(), 'target': y.cuda().long(), 'idxs': idx.cuda().long()}
    
    def __len__(self): 
        return len(self.dataloader)


def class_balance_dataset(dataset):
    min_label_freq = dataset[:][1].unique(return_counts=True)[1].min()
    selected_idxs = torch.LongTensor()
    for i in range(len(cifar_classes)):
        idxs = shuffle_tensor((dataset[:][1] == i).nonzero().squeeze())
        selected_idxs = torch.hstack((selected_idxs, idxs[:min_label_freq]))
    balanced_dataset = Subset(dataset, shuffle_tensor(selected_idxs))
    return balanced_dataset

def balance_dataset_w_class_imbalance(dataset, imbalance_class, class_imbalance_factor):
    if class_imbalance_factor == 1:
        return class_balance_dataset(dataset)
    imbalance_class_idx = cifar_classes.index(imbalance_class)
    label_freqs = dataset[:][1].unique(return_counts=True)[1]
    imbalance_class_freq = label_freqs[imbalance_class_idx] - label_freqs[imbalance_class_idx] % class_imbalance_factor
    other_class_freq = int(label_freqs[imbalance_class_idx] / class_imbalance_factor)
    selected_idxs = torch.LongTensor()
    for i in range(len(cifar_classes)):
        idxs = shuffle_tensor((dataset[:][1] == i).nonzero().squeeze())
        num_to_select = other_class_freq if i != imbalance_class_idx else imbalance_class_freq
        selected_idxs = torch.hstack((selected_idxs, idxs[:num_to_select]))
    balanced_dataset = Subset(dataset, shuffle_tensor(selected_idxs))
    return balanced_dataset

def rand_split_dataset(dataset, partition_size):
    rand_idx = torch.randperm(len(dataset))
    first_set = Subset(dataset, rand_idx[:partition_size])
    second_set = Subset(dataset, rand_idx[partition_size:])
    return first_set, second_set

def concat_dataset(dataset1, dataset2):
    return ConcatDataset([dataset1, dataset2])
