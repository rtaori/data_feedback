import argparse
from os.path import join
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Preprocess Cinic10 Dataset')
parser.add_argument('--data-dir', default='cinic10', type=str, help='path to existing cinic10 dataset')
parser.add_argument('--output-dir', default='cinic10', type=str, help='path to output directory')
args = parser.parse_args()


cinic_mean_RGB = (0.47889522, 0.47227842, 0.43047404)
cinic_std_RGB = (0.24205776, 0.23828046, 0.25874835)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cinic_mean_RGB, cinic_std_RGB),
])

train_set = torchvision.datasets.ImageFolder(root=join(args.data_dir, 'train'), transform=data_transform)
valid_set = torchvision.datasets.ImageFolder(root=join(args.data_dir, 'valid'), transform=data_transform)
test_set = torchvision.datasets.ImageFolder(root=join(args.data_dir, 'test'), transform=data_transform)

images = []
labels = []

for i, (image, label) in enumerate(tqdm(train_set)):
    images.append(image)
    labels.append(label)

for i, (image, label) in enumerate(tqdm(valid_set)):
    images.append(image)
    labels.append(label)

for i, (image, label) in enumerate(tqdm(test_set)):
    images.append(image)
    labels.append(label)

images = torch.stack(images)
labels = torch.tensor(labels)

torch.save(images, join(args.output_dir, 'images.pt'))
torch.save(labels, join(args.output_dir, 'labels.pt'))
