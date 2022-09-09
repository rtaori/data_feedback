import json
import argparse
from os.path import join
from tqdm import tqdm
import numpy as np
import wandb
import torch
from torch import optim

from imsitu import imSituTensorEvaluation, imSituVerbRoleLocalNounEncoder, imSituSituation
from baseline_crf import baseline_crf
import data


parser = argparse.ArgumentParser(description="Data Feedback on Visual Role-Labeling")
parser.add_argument('--wandb-log', action='store_true', help='if set, logs to wandb')
parser.add_argument('--wandb-group', type=str, default='debugging', help='group name for this experiment')
parser.add_argument('--data-dir', default='./data', type=str, help='path to data root')
parser.add_argument("--test-set-imgs-per-class", default=50, type=int)
parser.add_argument('-n', '--init-train-set-size', type=int, default=20000)
parser.add_argument('-m', '--num-human-labeled-samples', type=int, default=2500)
parser.add_argument('-k', '--num-model-labeled-samples', type=int, default=2500)
parser.add_argument('-a', "--arch", choices=["resnet_18", "resnet_34", "resnet_50", "resnet_101"], default="resnet_18")
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--learning-rate", default=1e-5, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)
parser.add_argument('--num-rounds', type=int, default=25)
parser.add_argument("--num-workers", default=4, type=int)
args = parser.parse_args()


def predict_human_readable(dataset_loader, encoding, model, top_k):
    model.eval()
    preds = {}
    top1 = imSituTensorEvaluation(1, 3, encoding)

    for ids, input, target in tqdm(dataset_loader):
        input_var = torch.autograd.Variable(input.cuda())
        (scores, predictions) = model.forward_max(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)
        top1.add_point(target, predictions.data, idx.data)

        human = encoding.to_situation(predictions)
        (b, p, d) = predictions.size()
        for _b in range(0, b):
            items = []
            offset = _b * p
            for _p in range(0, p):
                items.append(human[offset + _p])
                items[-1]["score"] = scores.data[_b][_p].item()
            items = sorted(items, key=lambda x: -x["score"])[:top_k]
            name = ids[_b].split(".")[:-1]
            preds[name[0]] = items

    return top1, preds


def train_model(max_epoch, train_loader, model, encoding, optimizer, n_refs=3):
    print('Training:')
    for _ in range(0, max_epoch):
        top1 = imSituTensorEvaluation(1, n_refs, encoding)
        loss_total = 0
        model.train()

        for _, (_, input, target) in enumerate(tqdm(train_loader)):
            input_var = torch.autograd.Variable(input.cuda())
            (_, v, vrn, norm, scores, predictions) = model(input_var)
            (_, idx) = torch.sort(scores, 1, True)

            optimizer.zero_grad()
            loss = model.mil_loss(v, vrn, norm, target, n_refs)
            loss.backward()

            optimizer.step()
            loss_total += loss.item()

            top1.add_point(target, predictions.data, idx.data)


def calculate_training_epochs(dataset_len):
    if dataset_len <= 20000:
        return 50
    elif 20000 < dataset_len <= 35000:
        return int(50 - 10 * (dataset_len - 20000) / 15000)
    elif 35000 < dataset_len <= 50000:
        return int(40 - 5 * (dataset_len - 35000) / 15000)
    elif 50000 < dataset_len <= 75000:
        return int(35 - 5 * (dataset_len - 50000) / 25000)
    elif 75000 < dataset_len:
        return 30


if __name__ == "__main__":
    wandb.init(project='data-feedback', group=args.wandb_group, mode='online' if args.wandb_log else 'disabled')
    wandb.config.update(vars(args))

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True

    # aggregate all datasets together
    train_set = json.load(open(join(args.data_dir, "imsitu/train.json")))
    dev_set = json.load(open(join(args.data_dir, "imsitu/dev.json")))
    test_set = json.load(open(join(args.data_dir, "imsitu/test.json")))
    dataset = train_set | dev_set | test_set

    # create imsitu mapping
    encoder = imSituVerbRoleLocalNounEncoder(dataset)

    # split off test & train sets, & collapse all training annotations into one per image (since we only have one pseudolabel)
    test_set, reserve_set = data.rand_split_test_set(dataset, args.test_set_imgs_per_class)
    reserve_set = data.collapse_annotations(reserve_set, use_majority=True)
    train_set, reserve_set = data.rand_split_dataset(reserve_set, args.init_train_set_size)

    for round in range(args.num_rounds):
        print(f'*** ROUND {round} ***')

        # initialize model and optimizer
        model = baseline_crf(encoder, cnn_type=args.arch, ngpus=1, prediction_type='max_max')
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        num_epochs = calculate_training_epochs(len(train_set))

        # create dataloaders
        dataset_train = imSituSituation(join(args.data_dir, 'of500_images_resized'), train_set, encoder, model.train_preprocess())
        dataset_test = imSituSituation(join(args.data_dir, 'of500_images_resized'), test_set, encoder, model.dev_preprocess())
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_model(num_epochs, train_loader, model, encoder, optimizer, n_refs=1)

        # evaluate model on test set and get results
        with torch.no_grad():
            top1, preds = predict_human_readable(test_loader, encoder, model, top_k=1)
        top1_a = top1.get_average_results()
        top1_a['avg-score'] = np.mean([v for v in top1_a.values()])

        # log eval results and other round stats
        stats = {'eval/'+k:v for k, v in top1_a.items()}        
        stats |= {'round': round, 'train_epochs': num_epochs, 'size_train_set': len(train_set), 'size_reserve_set': len(reserve_set)}

        # log gender bias metadata about the datasets and predictions
        preds_trans = data.transform_preds_to_dataset(preds)
        preds_stats = data.get_dataset_gender_stats(preds_trans)
        train_stats = data.get_dataset_gender_stats(train_set)
        test_stats = data.get_dataset_gender_stats(test_set)
        stats = stats | {'preds/'+k:v for k, v in preds_stats.items()}
        stats = stats | {'train/'+k:v for k, v in train_stats.items()}
        stats = stats | {'test/'+k:v for k, v in test_stats.items()}

        # log stats
        wandb.log(stats)

        if len(reserve_set) < args.num_human_labeled_samples + args.num_model_labeled_samples:
            print(f'Ending retraining - not enough remaining samples in reserve set ({len(reserve_set)} left).')
            break

        # add new human labeled samples per round
        if args.num_human_labeled_samples > 0:
            reserve_set_selected, reserve_set = data.rand_split_dataset(reserve_set, args.num_human_labeled_samples)
            train_set = train_set | reserve_set_selected

        # add new model labeled samples per round
        if args.num_model_labeled_samples > 0:
            reserve_set_partition, reserve_set = data.rand_split_dataset(reserve_set, args.num_model_labeled_samples)
            dataset_reserve = imSituSituation(join(args.data_dir, 'of500_images_resized'), reserve_set_partition, encoder, model.dev_preprocess())
            reserve_loader = torch.utils.data.DataLoader(dataset_reserve, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            # predict on unlabeled set
            with torch.no_grad():
                _, reserve_preds = predict_human_readable(reserve_loader, encoder, model, top_k=1)

            # pseudo-labeling and add to train set for next round
            reserve_preds_trans = data.transform_preds_to_dataset(reserve_preds)
            assert all(k in reserve_set_partition for k in reserve_preds_trans.keys())
            train_set = train_set | reserve_preds_trans
