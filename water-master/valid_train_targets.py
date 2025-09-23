import argparse
import collections
import torch
import torch.nn as nn
import numpy as np
import dataset
from dataset import ClimateDataset, ClimateDatasetTime, ClimateDatasetV2
from torch.utils.data import DataLoader
import mytransform
import model.loss as module_loss
import model.metric as module_metric
from model import model as module_model
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from predict import predict_whole_dataset

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config: ConfigParser):
    logger = config.get_logger('train')

    # setup data_loader instances
    # this mean and std is calculated on training data
    normalize = getattr(mytransform, config["normalize"])
    if config['normalize'] == "MinMaxNormalize":
        branch_transform = normalize(
            max=mytransform.BRANCH_MAX,
            min=mytransform.BRANCH_MIN
        )
        trunk_transform = normalize(
            max=mytransform.TRUNC_MAX,
            min=mytransform.TRUNC_MIN
        )
        if config['y_normalize']:
            target_transform = normalize(
                max=mytransform.TARGET_MAX,
                min=mytransform.TARGET_MIN,
            )
            target_detransform = mytransform.MinMaxDeNormalize(
                max=mytransform.TARGET_MAX,
                min=mytransform.TARGET_MIN,
            )
        else:
            target_transform = None
            target_detransform = None
    if config['normalize'] == "MeanStdNormalize":
        branch_transform = normalize(
            mean=mytransform.BRANCH_MEAN,
            std=mytransform.BRANCH_STD
        )
        trunk_transform = normalize(
            mean=mytransform.TRUNC_MEAN,
            std=mytransform.TRUNC_STD
        )
        if config['y_normalize']:
            target_transform = normalize(
                mean=mytransform.TARGET_MEAN,
                std=mytransform.TARGET_STD
            )
            target_detransform = mytransform.MeanStdDeNormalize(
                mean=mytransform.TARGET_MEAN,
                std=mytransform.TARGET_STD
            )
        else:
            target_transform = None
            target_detransform = None
    
    # target_transform = None
    if config['dataset'] == 'ClimateDataset':
        train_data = ClimateDataset("../Maumee DL/", branch_transform, trunk_transform, target_transform, 'train')
        val_data = ClimateDataset("../Maumee DL/", branch_transform, trunk_transform, target_transform, 'val')
    elif config['dataset'] == 'ClimateDatasetTime':
        train_data = ClimateDatasetTime("../Maumee DL/", branch_transform, trunk_transform, target_transform, 'train')
        val_data = ClimateDatasetTime("../Maumee DL/", branch_transform, trunk_transform, target_transform, 'test')
    elif config['dataset'] == 'ClimateDatasetV2':
        if config['normalize'] == "MinMaxNormalize":
            branch_transform = normalize(
                max=mytransform.BRANCH_MAXV2,
                min=mytransform.BRANCH_MINV2
            )
            trunk_transform = normalize(
                max=mytransform.TRUNC_MAXV2,
                min=mytransform.TRUNC_MINV2
            )
            if config['y_normalize']:
                target_transform = normalize(
                    max=mytransform.TARGET_MAXV2,
                    min=mytransform.TARGET_MINV2,
                )
                target_detransform = mytransform.MinMaxDeNormalize(
                    max=mytransform.TARGET_MAXV2,
                    min=mytransform.TARGET_MINV2,
                )
            else:
                target_transform = None
                target_detransform = None
        elif config['normalize'] == "MeanStdNormalize":
            branch_transform = normalize(
                mean=mytransform.BRANCH_MEANV2,
                std=mytransform.BRANCH_STDV2
            )
            trunk_transform = normalize(
                mean=mytransform.TRUNC_MEANV2,
                std=mytransform.TRUNC_STDV2
            )
            if config['y_normalize']:
                target_transform = normalize(
                    mean=mytransform.TARGET_MEANV2,
                    std=mytransform.TARGET_STDV2
                )
                target_detransform = mytransform.MeanStdDeNormalize(
                    mean=mytransform.TARGET_MEANV2,
                    std=mytransform.TARGET_STDV2
                )
            else:
                target_transform = None
                target_detransform = None
        
        train_data = ClimateDatasetV2("../climate_washed", branch_transform, trunk_transform, target_transform, 'train')
        val_data = ClimateDatasetV2("../climate_washed", branch_transform, trunk_transform, target_transform, 'test')
    else:
        data_class = getattr(dataset, config['dataset'])
        train_data = data_class("../climate_washed", config['normalize'], split="train", x_feature=config['x_feature'], y_feature=config['y_feature'])
        stats = train_data.get_stats()
        val_data = data_class("../climate_washed", config['normalize'], split="test", stats=stats,
                              x_feature=config['x_feature'], y_feature=config['y_feature'])
        target_detransform = train_data.get_target_detransform()

    data_loader = DataLoader(train_data, **config['data_loader']['args'], pin_memory=True)
    valid_data_loader = DataLoader(val_data, batch_size=2048, num_workers=10, pin_memory=True)
    
    valid_count = torch.zeros((2,))
    for _, _, target in data_loader:
        valid = ~torch.isnan(target)
        valid = torch.sum(valid, dim=0)
        valid_count += valid
    print(valid_count.tolist())



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
