""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy
import tensorflow as tf
from scipy.stats import ttest_ind
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

#TCAV部分的utils.py
def flatten(nested_list):
    """Flatten a nested list."""
    return [item for a_list in nested_list for item in a_list]


def process_what_to_run_expand(pairs_to_test,
                               random_counterpart=None,
                               num_random_exp=100,
                               random_concepts=None):
    """Get concept vs. random or random vs. random pairs to run.

      Given set of target, list of concept pairs, expand them to include
       random pairs. For instance [(t1, [c1, c2])...] becomes
       [(t1, [c1, random1],
        (t1, [c1, random2],...
        (t1, [c2, random1],
        (t1, [c2, random2],...]

    Args:
      pairs_to_test: [(target, [concept1, concept2,...]),...]
      random_counterpart: random concept that will be compared to the concept.
      num_random_exp: number of random experiments to run against each concept.
      random_concepts: A list of names of random concepts for the random
                       experiments to draw from. Optional, if not provided, the
                       names will be random500_{i} for i in num_random_exp.

    Returns:
      all_concepts: unique set of targets/concepts
      new_pairs_to_test: expanded
    """

    def get_random_concept(i):
        return (random_concepts[i] if random_concepts
                else 'random500_{}'.format(i))

    new_pairs_to_test = []
    for (target, concept_set) in pairs_to_test:
        new_pairs_to_test_t = []
        # if only one element was given, this is to test with random.
        if len(concept_set) == 1:
            i = 0
            while len(new_pairs_to_test_t) < min(100, num_random_exp):
                # make sure that we are not comparing the same thing to each other.
                if concept_set[0] != get_random_concept(
                        i) and random_counterpart != get_random_concept(i):
                    new_pairs_to_test_t.append(
                        (target, [concept_set[0], get_random_concept(i)]))
                i += 1
        elif len(concept_set) > 1:
            new_pairs_to_test_t.append((target, concept_set))
        else:
            tf.logging.info('PAIR NOT PROCCESSED')
        new_pairs_to_test.extend(new_pairs_to_test_t)

    all_concepts = list(set(flatten([cs + [tc] for tc, cs in new_pairs_to_test])))

    return all_concepts, new_pairs_to_test


def process_what_to_run_concepts(pairs_to_test):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])

    Returns:
      return pairs to test:
         target1, concept1
         target1, concept2
         ...
         target2, concept1
         target2, concept2
         ...

    """

    pairs_for_sstesting = []
    # prepare pairs for concpet vs random.
    for pair in pairs_to_test:
        for concept in pair[1]:
            pairs_for_sstesting.append([pair[0], [concept]])
    return pairs_for_sstesting


def process_what_to_run_randoms(pairs_to_test, random_counterpart):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])
      random_counterpart: a random concept that will be compared to the concept.

    Returns:
      return pairs to test:
            target1, random_counterpart,
            target2, random_counterpart,
            ...
    """
    # prepare pairs for random vs random.
    pairs_for_sstesting_random = []
    targets = list(set([pair[0] for pair in pairs_to_test]))
    for target in targets:
        pairs_for_sstesting_random.append([target, [random_counterpart]])
    return pairs_for_sstesting_random


# helper functions to write summary files
def print_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
                  min_p_val=0.05):
    """Helper function to organize results.
    If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
    If you get unexpected output, make sure you are using the correct keywords.

    Args:
      results: dictionary of results from TCAV runs.
      random_counterpart: name of the random_counterpart used, if it was used.
      random_concepts: list of random experiments that were run.
      num_random_exp: number of random experiments that were run.
      min_p_val: minimum p value for statistical significance
    """

    # helper function, returns if this is a random concept
    def is_random_concept(concept):
        if random_counterpart:
            return random_counterpart == concept

        elif random_concepts:
            return concept in random_concepts

        else:
            return 'random500_' in concept

    # print class, it will be the same for all
    print("Class =", results[0]['target_class'])

    # prepare data
    # dict with keys of concepts containing dict with bottlenecks
    result_summary = {}

    # random
    random_i_ups = {}

    for result in results:
        if result['cav_concept'] not in result_summary:
            result_summary[result['cav_concept']] = {}

        if result['bottleneck'] not in result_summary[result['cav_concept']]:
            result_summary[result['cav_concept']][result['bottleneck']] = []

        result_summary[result['cav_concept']][result['bottleneck']].append(result)

        # store random
        if is_random_concept(result['cav_concept']):
            if result['bottleneck'] not in random_i_ups:
                random_i_ups[result['bottleneck']] = []

            random_i_ups[result['bottleneck']].append(result['i_up'])

    # print concepts and classes with indentation
    for concept in result_summary:

        # if not random
        if not is_random_concept(concept):
            print(" ", "Concept =", concept)

            for bottleneck in result_summary[concept]:
                i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]

                # Calculate statistical significance
                _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)

                print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
                                                "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
                          bottleneck, numpy.mean(i_ups), numpy.std(i_ups),
                          numpy.mean(random_i_ups[bottleneck]),
                          numpy.std(random_i_ups[bottleneck]), p_val,
                          "not significant" if p_val > min_p_val else "significant"))


def make_dir_if_not_exists(directory):
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)