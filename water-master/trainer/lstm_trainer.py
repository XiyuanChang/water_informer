import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, remove_nan
import torch.nn.functional as F
import json
import datetime
from collections import defaultdict
from tqdm import tqdm
import copy
import model.metric as module_metric

class LSTMTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, valid_threshold=51, lr_scheduler=None, 
                 len_epoch=None, target_detransform=None, valid_data=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.valid_data = valid_data
        self.do_validation = self.valid_data_loader is not None or self.valid_data is not None
        # self.do_validation = False
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))*2
        self.log_epoch = config['trainer']['log_epoch']
        self.target_detransform = target_detransform
        self.rng = np.random.default_rng()

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.val_history = {'pbias': [], 'r_squared': [], 'kge': [], 'loss': [], 'per_target_loss': []}
        self.valid_threshold = valid_threshold

        num_class = len(self.config['y_feature']) if self.config['y_feature'] else 20
        self.best_target_loss = np.full(num_class, np.inf)
        self.best_target_state_dict = {k: None for k in range(num_class)}
        self.best_target_metric = {k: None for k in range(num_class)}
        self.num_class = num_class

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        print("Current time:", datetime.datetime.now().strftime("%H:%M:%S"))
        self.model.train()
        self.train_metrics.reset()
        outputs, targets = [], []

        for batch_idx, (x, y) in enumerate(self.data_loader):
            x, y = x.cuda(), y.cuda()
            self.optimizer.zero_grad()

            # self.criterion = torch.nn.MSELoss(reduction='none')
            output = self.model(x)
            
            # loss = self.criterion(output, target)
            threshold = self.config['threshold']
            
            # 6.23 original
            # loss_, _ = self.criterion(output, y, threshold=threshold)
            # loss = torch.mul(weighted_matrix, loss_).mean()
            # loss = loss_.mean()
            # loss.backward()
            
            _, loss_ = self.criterion(output, y, threshold=threshold)
            loss_backward = (loss_ / loss_.detach()).sum()
            loss_backward.backward()
            self.optimizer.step()
            loss = loss_.mean()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if epoch % self.log_epoch == 0:
                outputs.append(output[:, -1, :].detach().cpu())
                target = y[:, -1, :].cpu()
                target = torch.where(target == -1, torch.tensor(float('nan')), target)
                targets.append(target)

            if batch_idx == self.len_epoch:
                break
        
        if epoch % self.log_epoch == 0:
            outputs = torch.cat(outputs).cuda()
            targets = torch.cat(targets).cuda()
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(outputs, targets))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, val_raw = self._valid_epoch(epoch)
            val_update = {}
            
            for k, v in val_log.items():
                if isinstance(v, float):
                    self.val_history[k].append(v)
                else:
                    self.val_history[k].append(v.numpy())

                if k == 'pbias':
                    val_update['val_best_pbias'] = get_best_pbias(self.val_history[k])
                elif k == 'loss':
                    val_update['val_loss'] = v
                elif k == 'per_target_loss':
                    val_update['val_target_loss'] = v.tolist()
                # else:
                #     val_update[f'val_best_{k}'] = np.max(np.stack(self.val_history[k], axis=0), axis=0)
            
            loss_values = self.val_history['loss']
            lowest_index = np.argmin(loss_values)
            val_update['model_best_val_loss'] = self.val_history['loss'][lowest_index]
            val_update['model_best_r_squared'] = self.val_history['r_squared'][lowest_index]
            val_update['model_best_kge'] = self.val_history['kge'][lowest_index]
            val_update['model_best_pbias'] = self.val_history['pbias'][lowest_index]

            target_loss = 0 - val_log['kge']
            for i in range(self.num_class):
                if target_loss[i] < self.best_target_loss[i]:
                    print("Updating best model for target {}, {} -> {}".format(i, 0-self.best_target_loss[i].item(), 0-target_loss[i].item()))
                    self.best_target_loss[i] = target_loss[i]
                    self.best_target_state_dict[i] = copy.deepcopy(self.model.state_dict())
                    self.best_target_metric[i] = {
                        'pbias': val_log['pbias'].numpy(),
                        'r_squared': val_log['r_squared'].numpy(),
                        'kge': val_log['kge'].numpy()
                    }
                val_update[f'model_{i}_r_squared'] = self.best_target_metric[i]['r_squared']
                val_update[f'model_{i}_kge'] = self.best_target_metric[i]['kge']
                val_update[f'model_{i}_pbias'] = self.best_target_metric[i]['pbias']
            log.update(**{k : v for k, v in val_update.items()})
            
            # filename = str(self.checkpoint_dir / 'val_metric_epoch_{}.json'.format(epoch))

            # for k, v in val_raw.items():
            #     if isinstance(v[0], torch.Tensor):
            #         val_raw[k] = [x.tolist() for x in v]
            #     elif isinstance(v[0], np.ndarray):
            #         val_raw[k] = [x.tolist() for x in v]
            # json.dump(val_raw, open(filename, 'w'), indent=2)
            # self.logger.info('Saving validation results {}'.format(filename))

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_metrics.add_key('per_target_loss')
        if self.config['x_feature']:
            x_feat = len(self.config['x_feature'])
        else:
            x_feat = 26
        # outputs, targets = [], []
        if self.valid_data:
            for i in range(len(self.valid_data)):
                outputs, targets = [], []
                loss_per_sample_list, per_target_loss_list = [], []

                x, y = self.valid_data[i]
                x, y = x.to(self.device), y.to(self.device)
                
                for batch_idx in range(0, x.shape[0], 4096):
                    batch_x, batch_y = x[batch_idx:batch_idx+4096], y[batch_idx:batch_idx+4096]
                    pred = self.model(batch_x)
                    loss_per_sample, per_target_loss = self.criterion(pred, batch_y)
                    
                    pred = pred.detach().cpu()
                    pred_y = pred[:, -1, :]
                    target = batch_y[:, -1, :].cpu()
                    target = torch.where(target == -1, torch.tensor(float('nan')), target)
                    
                    outputs.append(pred_y)
                    targets.append(target)
                    loss_per_sample_list.append(loss_per_sample.detach().cpu())
                    per_target_loss_list.append(per_target_loss.detach().cpu())
                   
                    
                loss_per_sample = torch.cat(loss_per_sample_list, axis=0)
                per_target_loss = torch.stack(per_target_loss_list, axis=0).sum(axis=0)
                
                target = torch.cat(targets, axis=0)
                pred_y = torch.cat(outputs, axis=0)
                # output = self.model(x)
                # loss_per_sample, per_target_loss = self.criterion(output, y)

                loss = loss_per_sample.mean()
                # loss = per_target_loss.mean()
                # output = output.detach()
                # pred_y = output[:, -1, :]
                # target = y[:, -1, :]
                # target = torch.where(target == -1, torch.tensor(float('nan')), target)

                if self.target_detransform:
                    pred_y = self.target_detransform(pred_y)
                    target = self.target_detransform(target)
                print("target shape:", target.shape)

                valid_count = (~torch.isnan(target)).sum(axis=0).cpu()
            
                # import pdb; pdb.set_trace()
                per_target_loss = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), per_target_loss)
                # if i == 50:
                    # print(valid_count)
                    # break

                # outputs.append(output.detach())
                # targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data) + i, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('per_target_loss', per_target_loss)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_y, target), n=1)
                    # change the 30 below to something else
                    self.valid_metrics._data[met.__name__][-1] = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), self.valid_metrics._data[met.__name__][-1])

            # print(self.valid_metrics._data['nse'])
            # all = self.valid_metrics._data['nse']
            # all = torch.stack(all,axis=0).numpy()
            # import pdb;pdb.set_trace()

        else:
            targets = defaultdict(list)
            predictions = defaultdict(list)
            masks = defaultdict(list)
            print("Validating...")

            with torch.no_grad():
                for batch_idx, (staids, x, y) in enumerate(tqdm(self.valid_data_loader)):
                    x, y = x.cuda(), y.cuda()

                    output = self.model(x)
                    loss_per_sample, _ = self.criterion(output, y)
                    loss = torch.nanmean(loss_per_sample)
                    # loss = loss_per_sample.mean()
                    # per_target_loss = per_target_loss.detach().cpu()
                    
                    pred = output.detach()
                    pred = pred[:, -1, :].cpu()
                    target = y[:, -1, :].cpu()
                    
                    # x = x[:, -1, :26].cpu()
                    x = x[:, -1, :x_feat].cpu()
                    mask = ~(x == -1).any(dim=1)
                    
                    # import pdb; pdb.set_trace()
                    # target = torch.where(target == -1, torch.tensor(float('nan')), target)

                    # if self.target_detransform:
                        # pred = self.target_detransform(pred).cpu()
                        # target = self.target_detransform(target).cpu()
                    
                    for idx, staid in enumerate(staids):
                        targets[staid].append(target[idx])
                        predictions[staid].append(pred[idx])
                        masks[staid].append(mask[idx])

                    # valid_count = (~torch.isnan(target)).sum(axis=0).cpu()
                    # per_target_loss = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), per_target_loss)
                    
                    self.valid_metrics.update('loss', loss.item())
                    # self.valid_metrics.update('per_target_loss', per_target_loss)
                    
                    # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            for staid in targets.keys():
                # import pdb; pdb.set_trace()
                target = torch.stack(targets[staid], axis=0).cuda()
                pred = torch.stack(predictions[staid], axis=0).cuda()
                mask = torch.stack(masks[staid], axis=0).cuda()

                target = target[mask]
                pred = pred[mask]
                if len(target) == 0:
                    continue
                
                # loss_per_target = mse_val(pred, target).cpu()
                # target = torch.where(target == -1, torch.tensor(float('nan')), target)
                
                pred = self.target_detransform(pred).cpu()
                target = self.target_detransform(target).cpu()
                
                loss_per_target = 0-module_metric.r_squared(pred, target)
                valid_count = (~torch.isnan(target)).sum(axis=0).cpu()
                loss_per_target = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), loss_per_target)
                
                self.valid_metrics.update('per_target_loss', loss_per_target)

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred, target), n=1)
                    self.valid_metrics._data[met.__name__][-1] = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), self.valid_metrics._data[met.__name__][-1])

        return self.valid_metrics.result(raw=True)
    
    def post_training(self):
        filepath = str(self.checkpoint_dir / 'best_target_state_dict.pth')
        torch.save(self.best_target_state_dict, filepath)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

def gaussian_kernel_density_estimation(input_vector, sigma=1):
    """
    Perform Gaussian kernel density estimation on a tensor.

    Parameters:
    input_vector (torch.Tensor): Input tensor of shape [size, feature_num].
    sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
    torch.Tensor: Output tensor of shape [size, size].
    """
    distances = torch.norm(input_vector.unsqueeze(1) - input_vector, dim=2)
    output = torch.exp(-distances**2 / sigma)
    output /= torch.sum(output, dim=1, keepdim=True)

    return output

def get_mixup_sample_rate(data, sigma=1):
    if data.dim() == 1:
        data = data.unsqueeze(1)
    
    return gaussian_kernel_density_estimation(data, sigma)


def knn_regression(train_features, train_targets, test_features, k):
    """
    Perform k-NN regression using PyTorch tensors.

    :param train_features: A tensor of shape (num_train_samples, num_features) containing the training data features.
    :param train_targets: A tensor of shape (num_train_samples,) or (num_train_samples, 1) containing the training data targets.
    :param test_features: A tensor of shape (num_test_samples, num_features) containing the test data features.
    :param k: The number of nearest neighbors to consider for regression.
    :return: Predicted targets for the test data.
    """
    # Ensure the input tensors are on the GPU
    train_features = train_features.cuda()
    train_targets = train_targets.cuda()
    test_features = test_features.cuda()

    # Calculate the pairwise distances between test points and all training points
    distances = torch.norm(train_features.unsqueeze(1) - test_features.unsqueeze(0), dim=2)

    # Determine the indices of the k nearest neighbors (using topk to find the k smallest distances)
    _, indices = torch.topk(distances, k, largest=False, sorted=False)

    # Gather the targets of the k nearest neighbors
    nearest_targets = torch.gather(train_targets.expand(-1, k), 0, indices)

    # Calculate the mean target value of the k nearest neighbors
    predictions = nearest_targets.mean(dim=1)

    return predictions

def get_best_pbias(arr):

    array = np.array(arr)
    abs_arr = np.abs(array)
    
    # Find the index of the minimum absolute value in each column
    min_index = np.argmin(abs_arr, axis=0)
    
    # Select the elements from each column based on these indices
    min_abs_values = array[min_index, np.arange(array.shape[1])]
    
    return min_abs_values

def mse_val(input, target):
    # Ensure the target is a float tensor (to handle NaN)
    target = target.float()
    
    # Create a mask that will be True where target is not NaN
    # mask = target != -1
    mask = ~torch.isnan(target)
    
    # Apply the mask to both input and target to ignore NaNs in target
    filtered_input = torch.where(mask, input, torch.tensor(0.0, device=input.device))
    filtered_target = torch.where(mask, target, torch.tensor(0.0, device=target.device))
    
    # Compute the squared differences; masked out values will contribute 0 to the sum
    diff = filtered_input - filtered_target
    square_diff = diff ** 2
    
    valid_targets = mask.sum(dim=0).float()
    valid_targets = torch.where(valid_targets > 0, valid_targets, torch.tensor(1.0, device=target.device))
    loss_per_target = square_diff.sum(dim=0) / valid_targets

    return loss_per_target