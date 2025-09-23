from model.model import LSTM
from parse_config import ConfigParser
import argparse
import collections
import lstm_dataset as dataset
import model.lstm_loss as module_loss
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import MetricTracker
from trainer import LSTMTrainer
import model.metric as module_metric
from utils import prepare_device
import torch
import numpy as np
from lstm_predict import predict_whole_dataset
import warnings

warnings.filterwarnings("ignore")

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

def main(config: ConfigParser):
    logger = config.get_logger('train')
    assert config['dataset'] in ['ClimateDatasetV2A', 'ClimateDatasetV2B', 'ClimateDatasetV2C', 'ClimateDatasetV2D'], "Dataset still not supported"

    model = LSTM(**config['arch'])
    logger.info(model)
    logger.info(config.config)
    
    data_class = getattr(dataset, config['dataset'])
    root_dir = config['root_dir'] if config['root_dir'] is not None else "../climate_new"
    train_data = data_class(root_dir, config['normalize'], split="train", x_feature=config['x_feature'], 
                            y_feature=config['y_feature'], exclude=config['exclude'], minmax_feature=config['minmax_feature'])
    stats = train_data.get_stats()
    val_data = data_class(root_dir, config['normalize'], split="testsubset", stats=stats,
                          testNum=config['test_station'], x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'],
                          minmax_feature=config['minmax_feature'])
    target_detransform = train_data.get_target_detransform()
    
    batch_size = config['data_loader']['args']['batch_size']
    nDay = int(13514 * 0.8)
    nStation = 482
    nIter =  int(
        np.ceil(np.log(0.01) / np.log(1 - batch_size * 365 / nStation / nDay))
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    train_sampler = WeightedRandomSampler(torch.ones(len(train_data)), replacement=True, num_samples=nIter*batch_size)
    data_loader = DataLoader(train_data, **config['data_loader']['args'], pin_memory=True, sampler=train_sampler)
    valid_data_loader = DataLoader(val_data, batch_size=1024, num_workers=12, pin_memory=True, shuffle=False)

    # prepare for multi-device training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.cuda()
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    criterion = module_loss.mse_loss
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = LSTMTrainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                        #   valid_data=val_data,
                          lr_scheduler=lr_scheduler,
                          valid_threshold=config['valid_threshold'],
                          target_detransform=target_detransform,
                          )

    trainer.train()
    trainer.post_training()
    best_states = trainer.best_target_state_dict
    
    model_states = []
    for s in best_states.values():
        model_states.append(s)
    split_dict = {
        "ClimateDatasetV2A": "split_datesA.txt",
        "ClimateDatasetV2B": "split_datesB.txt",
        "ClimateDatasetV2C": "split_datesC.txt",
        "ClimateDatasetV2D": "station",
    }
    print("Saving prediction results to: ", str(trainer.checkpoint_dir / "predictions"))
    predict_whole_dataset(
        model,
        model_states,
        stats,
        config,
        valid_threshold=config['metric_threshold'],
        normalize=config['normalize'],
        split=split_dict[config['dataset']],
        root="../climate_new",
        output_dir=str(trainer.checkpoint_dir / "predictions"),
        x_feature=config['x_feature'],
        y_feature=config['y_feature'],
        exclude=config['exclude'],
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('-c', '--config', default='config_LSTM.json', type=str,
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
