import argparse
import wandb
import torch

from baidunet import BaiduNet9Network, BaiduNetOptimizer
from resnet import ResNet18Network, ResNetOptimizer
from data import Batches, cifar_classes, get_transforms, load_data, \
                 rand_split_dataset, concat_dataset, balance_dataset_w_class_imbalance
from utils import Timer, TableLogger, StatsLogger


parser = argparse.ArgumentParser(description='Data Feedback on Image Classification')
parser.add_argument('--wandb-log', action='store_true', help='if set, logs to wandb')
parser.add_argument('--wandb-group', type=str, default='debugging', help='group name for this experiment')
parser.add_argument('-d', '--dataset', choices=['cinic10', 'cifar5m'], default='cifar5m')
parser.add_argument('--data-dir', default='./data', type=str, help='path to data root')
parser.add_argument('--test-set-size', type=int, default=50000)
parser.add_argument('-n', '--init-train-set-size', type=int, default=50000)
parser.add_argument('-m', '--num-human-labeled-samples', type=int, default=2500)
parser.add_argument('-k', '--num-model-labeled-samples', type=int, default=2500)
parser.add_argument('--class-imbalance-class', choices=cifar_classes, default='dog')
parser.add_argument('--class-imbalance-factor', type=int, default=9, help='imbalances the class in a factor:1 ratio')
parser.add_argument('-a', '--arch', choices=['baidunet9', 'resnet18'], default='baidunet9')
parser.add_argument('--underfit-model', action='store_true') # trains model for only 5 epochs
parser.add_argument('--subsample-train-set-each-round', action='store_true')
parser.add_argument('--num-rounds', type=int, default=90)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--num-workers', type=int, default=6)
args = parser.parse_args()


def run_batches(model, batches, optimizer):
    stats = StatsLogger(('loss', 'correct'))
    model.train(True)
    torch.set_grad_enabled(True)
    for batch in batches:
        output = model(batch)
        stats.append(output)
        output['loss'].sum().backward()
        optimizer.step()
        model.zero_grad()
    return stats


def train(model, optimizer, train_batches, epochs, logger, timer, batch_size):  
    for epoch in range(epochs):
        train_stats, train_time = run_batches(model, train_batches, optimizer), timer()
        epoch_stats = { 
            'train_time': train_time, 'train_loss': train_stats.mean('loss'), 'train_acc': train_stats.mean('correct'), 
            'total_time': timer.total_time, 'epoch': epoch, 'lr': optimizer.param_values()['lr']*batch_size,
        }
        logger.append(epoch_stats)
        epoch_stats = {'train_stats/'+k: v for k, v in epoch_stats.items()}
        wandb.log(epoch_stats)
    return epoch_stats


def calc_test_stats(targs, preds, imbalance_class):
    accuracy = (preds == targs).float().mean()
    if imbalance_class is not None:
        imbalance_class_idx = cifar_classes.index(imbalance_class)
        frac_imbalance_class_pred = (preds == imbalance_class_idx).float().mean()
        frac_imbalance_class_targ = (targs == imbalance_class_idx).float().mean()
        frac_imbalance_class_bias = torch.abs(frac_imbalance_class_pred - frac_imbalance_class_targ)

    return_dict = {'test_stats/accuracy': accuracy}
    if imbalance_class is not None:
        return_dict |= {'test_stats/frac_imbalance_class_bias': frac_imbalance_class_bias,
                        'test_stats/frac_imbalance_class_pred': frac_imbalance_class_pred, 
                        'test_stats/frac_imbalance_class_targ': frac_imbalance_class_targ}
    return return_dict


def test(model, batches):
    model.eval()
    torch.set_grad_enabled(False)
    root_idxs = torch.cuda.LongTensor()
    targs = torch.cuda.LongTensor()
    preds = torch.cuda.LongTensor()
    for batch in batches:
        output = model(batch)
        root_idxs = torch.hstack((root_idxs, output['idxs']))
        targs = torch.hstack((targs, output['target']))
        preds = torch.hstack((preds, output['classifier'].max(dim=1).indices))
    return root_idxs, targs, preds


def base_training_epochs(num_examples):
    if num_examples <= 20000:
        epochs = 25
    elif 20000 < num_examples <= 50000:
        epochs = 25 - 5 * (num_examples - 20000) / 30000
    elif 50000 < num_examples <= 100000:
        epochs = 20 - 5 * (num_examples - 50000) / 50000
    elif 100000 < num_examples <= 1000000:
        epochs = 15 - 5 * (num_examples - 100000) / 900000
    elif 1000000 < num_examples:
        epochs = 10
    return int(epochs * 2.5) # correction factor to make sure network is properly trained


if __name__ == "__main__":
    wandb.init(project='data-feedback', group=args.wandb_group, mode='online' if args.wandb_log else 'disabled')
    wandb.config.update(vars(args))

    torch.backends.cudnn.benchmark = True
    timer = Timer()

    # load dataset, class imbalance it, split into initial train/test sets
    tensor_dataset = load_data(args.dataset, args.data_dir)
    train_transform, test_transform = get_transforms(args.dataset)
    tensor_dataset = balance_dataset_w_class_imbalance(tensor_dataset, args.class_imbalance_class, args.class_imbalance_factor)
    test_set, reserve_set = rand_split_dataset(tensor_dataset, args.test_set_size)
    train_set, reserve_set = rand_split_dataset(reserve_set, args.init_train_set_size)
    test_batches = Batches(test_set, test_transform, args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    for round in range(args.num_rounds):
        print(f'*** ROUND {round} ***')
        
        # subsample train set to initial size if requested
        curr_train_set = train_set
        if args.subsample_train_set_each_round:
            curr_train_set, _ = rand_split_dataset(curr_train_set, args.init_train_set_size)
        train_batches = Batches(curr_train_set, train_transform, args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)

        # create model, optimizer, and training schedule
        if args.arch == 'baidunet9':
            model = BaiduNet9Network()
            if args.underfit_model:
                num_epochs = 5
                opt = BaiduNetOptimizer(model, torch.optim.SGD, momentum=0.9, weight_decay=5e-4*args.batch_size, dampening=0, nesterov=True,
                                        lr_knots=[0*len(train_batches), 3*len(train_batches), 4*len(train_batches), 5*len(train_batches)], 
                                        lr_vals=[0.1/args.batch_size, 0.1/args.batch_size, 0.01/args.batch_size, 0.001/args.batch_size])
            else:
                num_epochs = base_training_epochs(len(curr_train_set))
                opt = BaiduNetOptimizer(model, torch.optim.SGD, momentum=0.9, weight_decay=5e-4*args.batch_size, dampening=0, nesterov=True,
                                    lr_knots=[0*len(train_batches), int(num_epochs/5)*len(train_batches), num_epochs*len(train_batches)], 
                                    lr_vals=[0/args.batch_size, 0.4/args.batch_size, 0.001/args.batch_size]) 
        elif args.arch == 'resnet18':
            model = ResNet18Network()
            if args.underfit_model:
                num_epochs = 5
            else:
                num_epochs = base_training_epochs(len(curr_train_set)) * 2 # resnet18 takes longer to train
            opt = ResNetOptimizer(model, lr=0.1/args.batch_size, T_max=num_epochs*len(train_batches))

        # train and record test-time statistics
        train(model, opt, train_batches, num_epochs, TableLogger(), timer, args.batch_size)
        _, targs, preds = test(model, test_batches)
        stats = calc_test_stats(targs, preds, args.class_imbalance_class)
        stats |= {'round': round, 'train_epochs': num_epochs, 'size_train_set': len(train_set), 'size_reserve_set': len(reserve_set)}
        wandb.log(stats)

        if len(reserve_set) < args.num_human_labeled_samples + args.num_model_labeled_samples:
            print(f'Ending retraining - not enough remaining samples in reserve set ({len(reserve_set)} left).')
            break
        
        # add new human labeled samples per round
        if args.num_human_labeled_samples > 0:
            reserve_set_selected, reserve_set = rand_split_dataset(reserve_set, args.num_human_labeled_samples)
            train_set = concat_dataset(train_set, reserve_set_selected)
        
        # add new model labeled samples per round
        if args.num_model_labeled_samples > 0:
            reserve_set_partition, reserve_set = rand_split_dataset(reserve_set, args.num_model_labeled_samples)
            reserve_batches = Batches(reserve_set_partition, test_transform, args.batch_size, 
                                      num_workers=args.num_workers, shuffle=False, drop_last=False)
            root_idxs, targs, preds = test(model, reserve_batches)
            assert torch.equal(reserve_set_partition[:][2].sort().values, root_idxs.sort().values.cpu())
            tensor_dataset.dataset.tensors[1][root_idxs] = preds.cpu() # pseudo labeling
            train_set = concat_dataset(train_set, reserve_set_partition)
