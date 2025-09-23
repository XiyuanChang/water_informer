import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn.functional as F
import json
import copy
import model.loss as module_loss
import model.metric as module_metric


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, valid_threshold=51, lr_scheduler=None, 
                 len_epoch=None, target_detransform=None, valid_data=None, num_class=20):
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
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))*2
        self.log_epoch = config['trainer']['log_epoch']
        self.target_detransform = target_detransform
        self.rng = np.random.default_rng()

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.val_history = {'pbias': [], 'r_squared': [], 'kge': [], 'r': [], 'beta': [], 'alpha': [], 'nse': [], 'loss': [], 'per_target_loss': []}
        self.val_history = {'pbias': [], 'r_squared': [], 'kge': [], 'loss': [], 'per_target_loss': []}
        self.valid_threshold = valid_threshold

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
        self.model.train()
        self.train_metrics.reset()
        outputs, targets = [], []

        for batch_idx, (trunk, branch, target) in enumerate(self.data_loader):
            trunk, branch, target = trunk.to(self.device), branch.to(self.device), target.to(self.device)
            if torch.isnan(trunk).any():
                raise ValueError("Trunk contains NaN")
            if torch.isnan(branch).any():
                raise ValueError("Branch contains NaN")
            self.optimizer.zero_grad()
            
            
            if self.config['mixup']:
                sample_rate = get_mixup_sample_rate(target, sigma=self.config['sigma'])
                lamda = np.random.beta(self.config['mix_alpha'], self.config['mix_beta'])

                # sample index based on sample rate
                cumsum = sample_rate.cumsum(1)
                randvec = torch.rand(sample_rate.shape[0], 1, device=sample_rate.device)
                bool_tensor = (cumsum > randvec).float()
                idx = bool_tensor.argmax(1)
                # idx = (sample_rate.cumsum(1) > torch.rand(sample_rate.shape[0], 1, device=sample_rate.device)).float().argmax(1)
                # idx = np.array(
                #     [self.rng.choice(np.arange(trunk.shape[0]), p=sample_rate[sel_idx]) for sel_idx in
                #     range(trunk.shape[0])])
                
                x1, u1, y1 = trunk, branch, target
                x2, u2, y2 = trunk[idx], branch[idx], target[idx]
                mixup_x = lamda * x1 + (1 - lamda) * x2
                mixup_u = lamda * u1 + (1 - lamda) * u2
                
                y1_2 = knn_regression(torch.cat([trunk, branch], dim=1), target, torch.cat([x1, u2], dim=1), self.config['k']).unsqueeze(1)
                y2_1 = knn_regression(torch.cat([trunk, branch], dim=1), target, torch.cat([x2, u1], dim=1), self.config['k']).unsqueeze(1)
                mixup_y = lamda ** 2 * y1 + (1 - lamda) ** 2 * y2 + lamda * (1 - lamda) * (y1_2 + y2_1)
                # import pdb;pdb.set_trace()
                output = self.model(mixup_x, mixup_u)
                loss, _ = self.criterion(output, mixup_y)
                loss.backward()
                self.optimizer.step()                
            else:
                # self.criterion = torch.nn.MSELoss(reduction='none')
                output = self.model(trunk, branch)
                # if epoch == 2:
                    # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                
                # loss = self.criterion(output, target)
                threshold = self.config['threshold']
                # weight = torch.tensor([1, 1])
                # 6.23 original
                # loss_, _ = self.criterion(output, target, threshold=threshold)
                # loss = loss_.mean()
                # loss.backward()
                _, loss_ = self.criterion(output, target, threshold)
                loss_backward = (loss_ / loss_.detach()).sum()
                loss_backward.backward()
                loss = loss_.mean()
                
                # loss = torch.mul(weighted_matrix, loss_).mean()
                
                self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            if epoch % self.log_epoch == 0:
                outputs.append(output.detach())
                targets.append(target.detach())

            if batch_idx == self.len_epoch:
                break
        
        if epoch % self.log_epoch == 0:
            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
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
                # elif k == 'per_target_loss':
                #     val_update['val_target_loss'] = v.tolist()
                # else:
                #     val_update[f'val_best_{k}'] = np.max(np.stack(self.val_history[k], axis=0), axis=0)

            # loss_values = np.array(self.val_history['per_target_loss'])
            # min_idxs = np.argmin(loss_values, axis=0)
            # val_update['this_r_squared'] = val_log['r_squared'].tolist()
            
            # for num, idx in enumerate(min_idxs):
            #     val_update[f'model_{num}_r_squared'] = self.val_history['r_squared'][idx]
            #     val_update[f'model_{num}_kge'] = self.val_history['kge'][idx]
            #     val_update[f'model_{num}_pbias'] = self.val_history['pbias'][idx]
            
            loss_values = self.val_history['loss']
            lowest_index = np.argmin(loss_values)
            val_update['model_best_val_loss'] = self.val_history['loss'][lowest_index]
            val_update['model_best_r_squared'] = self.val_history['r_squared'][lowest_index]
            val_update['model_best_kge'] = self.val_history['kge'][lowest_index]
            val_update['model_best_pbias'] = self.val_history['pbias'][lowest_index]
            # val_update['model_best_r'] = self.val_history['r'][lowest_index]
            # val_update['beta'] = self.val_history['beta'][lowest_index]
            # val_update['alpha'] = self.val_history['alpha'][lowest_index]
            # val_update['nse'] = self.val_history['nse'][lowest_index]
            
            # if lowest_index == len(loss_values) - 1:
            #     # save this best model
            #     print("Saving best model to ", str(self.checkpoint_dir / 'model_best.pth'))
            #     torch.save(self.model.state_dict(), str(self.checkpoint_dir / 'model_best.pth'))
            
            target_loss = 0 - val_log['kge']
            # target_loss = val_log['loss']
            # print(target_loss.tolist())
            for i in range(self.num_class):
                if target_loss[i] < self.best_target_loss[i]:
                    print("Updating best model for target {}, {} -> {}".format(i, 0-self.best_target_loss[i].item(), 0-target_loss[i].item()))
                    self.best_target_loss[i] = target_loss[i]
                    self.best_target_state_dict[i] = copy.deepcopy(self.model.state_dict())
                    self.best_target_metric[i] = {
                        'pbias': val_log['pbias'].numpy(),
                        'r_squared': val_log['r_squared'].numpy(),
                        'kge': val_log['kge'].numpy()
                        # 'r': val_log['r'].numpy(),
                        # 'beta': val_log['beta'].numpy(),
                        # 'alpha': val_log['alpha'].numpy(),
                        # 'nse': val_log['nse'].numpy()
                    }
                val_update[f'model_{i}_r_squared'] = self.best_target_metric[i]['r_squared']
                val_update[f'model_{i}_kge'] = self.best_target_metric[i]['kge']
                val_update[f'model_{i}_pbias'] = self.best_target_metric[i]['pbias']
                # val_update[f'model_{i}_r'] = self.best_target_metric[i]['r']
                # val_update[f'model_{i}_beta'] = self.best_target_metric[i]['beta']
                # val_update[f'model_{i}_alpha'] = self.best_target_metric[i]['alpha']
                # val_update[f'model_{i}_nse'] = self.best_target_metric[i]['nse']
                
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
        outputs, targets = [], []
        if self.valid_data:
            for i in range(len(self.valid_data)):
                # outputs, targets = [], []
                trunk, branch, target = self.valid_data[i]
                trunk, branch, target = trunk.to(self.device), branch.to(self.device), target.to(self.device)
                # import pdb; pdb.set_trace()

                output = self.model(trunk, branch)
                
                # 这行用来计算反归一化前每个y分量的loss，per_target_loss用于选样本
                # criterion用法：loss_per_sample, per_target_loss = self.criterion(output, target)
                # @JL: XB revised this part
                # _, per_target_loss = self.criterion(output, target)
                # print(self.target_detransform(output).shape)
                output = output.detach()
                per_target_loss = -1 * module_metric.r_squared(self.target_detransform(output), self.target_detransform(target))

                # loss = loss_per_sample.mean()
                if self.target_detransform:
                    output = self.target_detransform(output)
                    target = self.target_detransform(target)
                
                # 这行用来计算反归一化后每个样本的loss，loss_per_sample用于report val loss
                # loss_per_sample, _ = self.criterion(output, target, weight=torch.tensor([1, 1]))
                loss_per_sample, _ = self.criterion(output, target)
                loss = loss_per_sample.mean()
                # if epoch == 100:
                #     print(loss.item())

                valid_count = (~torch.isnan(target)).sum(axis=0).cpu()
                per_target_loss = per_target_loss.detach().cpu()
                per_target_loss = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), per_target_loss)
                # if i == 103:
                    # print(valid_count)

                # outputs.append(output.detach())
                # targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data) + i, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('per_target_loss', per_target_loss)
                for met in self.metric_ftns:
                    metric_value = met(output, target)
                    metric_value = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), metric_value)
                    self.valid_metrics.update(met.__name__, metric_value)
                    # change the 30 below to something else
                    # self.valid_metrics._data[met.__name__][-1] = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), self.valid_metrics._data[met.__name__][-1])

            # print(self.valid_metrics._data['nse'])
            # all = self.valid_metrics._data['nse']
            # all = torch.stack(all,axis=0).numpy()
            # import pdb;pdb.set_trace()

        else:
            with torch.no_grad():
                for batch_idx, (trunk, branch, target) in enumerate(self.valid_data_loader):
                    trunk, branch, target = trunk.to(self.device), branch.to(self.device), target.to(self.device)

                    output = self.model(trunk, branch)
                    # loss, _ = self.criterion(output, target, weight=torch.tensor([1, 1]))
                    loss, _ = self.criterion(output, target)
                    # loss = criterion(output, target)
                    # loss = loss.mean()

                    outputs.append(output.detach())
                    targets.append(target.detach())

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    # self.valid_metrics.update('loss', loss.item())
                    
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
            if self.target_detransform:
                outputs = self.target_detransform(outputs)
                targets = self.target_detransform(targets)
            # loss, _ = self.criterion(outputs, targets, weight=torch.tensor([1, 1]))
            loss, _ = self.criterion(outputs, targets)
            self.valid_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
            # self.writer.add_histogram(name, p, bins='auto')
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

class TrainerOne(Trainer):
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        outputs, targets = [], []

        for batch_idx, (trunk, branch, target) in enumerate(self.data_loader):
            trunk, branch, target = trunk.to(self.device), branch.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            x = torch.concat([trunk, branch], dim=1)
            
            # self.criterion = torch.nn.MSELoss(reduction='none')
            output = self.model(x).squeeze()

            # loss = self.criterion(output, target)
            threshold = self.config['threshold']
            # weight = torch.tensor([1, 1])
            # 6.23 original
            # loss_, _ = self.criterion(output, target, threshold=threshold)
            # loss = loss_.mean()
            # loss.backward()
            _, loss_ = self.criterion(output, target, threshold)
            loss_backward = (loss_ / loss_.detach()).sum()
            loss_backward.backward()
            loss = loss_.mean()
            
            # loss = torch.mul(weighted_matrix, loss_).mean()
            
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            if epoch % self.log_epoch == 0:
                outputs.append(output.detach())
                targets.append(target.detach())

            if batch_idx == self.len_epoch:
                break
        
        if epoch % self.log_epoch == 0:
            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
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
                # elif k == 'per_target_loss':
                #     val_update['val_target_loss'] = v.tolist()
                # else:
                #     val_update[f'val_best_{k}'] = np.max(np.stack(self.val_history[k], axis=0), axis=0)

            # loss_values = np.array(self.val_history['per_target_loss'])
            # min_idxs = np.argmin(loss_values, axis=0)
            # val_update['this_r_squared'] = val_log['r_squared'].tolist()
            
            # for num, idx in enumerate(min_idxs):
            #     val_update[f'model_{num}_r_squared'] = self.val_history['r_squared'][idx]
            #     val_update[f'model_{num}_kge'] = self.val_history['kge'][idx]
            #     val_update[f'model_{num}_pbias'] = self.val_history['pbias'][idx]
            
            loss_values = self.val_history['loss']
            lowest_index = np.argmin(loss_values)
            val_update['model_best_val_loss'] = self.val_history['loss'][lowest_index]
            val_update['model_best_r_squared'] = self.val_history['r_squared'][lowest_index]
            val_update['model_best_kge'] = self.val_history['kge'][lowest_index]
            val_update['model_best_pbias'] = self.val_history['pbias'][lowest_index]
            # val_update['model_best_r'] = self.val_history['r'][lowest_index]
            # val_update['beta'] = self.val_history['beta'][lowest_index]
            # val_update['alpha'] = self.val_history['alpha'][lowest_index]
            # val_update['nse'] = self.val_history['nse'][lowest_index]
            
            # if lowest_index == len(loss_values) - 1:
            #     # save this best model
            #     print("Saving best model to ", str(self.checkpoint_dir / 'model_best.pth'))
            #     torch.save(self.model.state_dict(), str(self.checkpoint_dir / 'model_best.pth'))
            
            target_loss = 0 - val_log['kge']
            # target_loss = val_log['loss']
            # print(target_loss.tolist())
            for i in range(self.num_class):
                if target_loss[i] < self.best_target_loss[i]:
                    print("Updating best model for target {}, {} -> {}".format(i, 0-self.best_target_loss[i].item(), 0-target_loss[i].item()))
                    self.best_target_loss[i] = target_loss[i]
                    self.best_target_state_dict[i] = copy.deepcopy(self.model.state_dict())
                    self.best_target_metric[i] = {
                        'pbias': val_log['pbias'].numpy(),
                        'r_squared': val_log['r_squared'].numpy(),
                        'kge': val_log['kge'].numpy()
                        # 'r': val_log['r'].numpy(),
                        # 'beta': val_log['beta'].numpy(),
                        # 'alpha': val_log['alpha'].numpy(),
                        # 'nse': val_log['nse'].numpy()
                    }
                val_update[f'model_{i}_r_squared'] = self.best_target_metric[i]['r_squared']
                val_update[f'model_{i}_kge'] = self.best_target_metric[i]['kge']
                val_update[f'model_{i}_pbias'] = self.best_target_metric[i]['pbias']
                # val_update[f'model_{i}_r'] = self.best_target_metric[i]['r']
                # val_update[f'model_{i}_beta'] = self.best_target_metric[i]['beta']
                # val_update[f'model_{i}_alpha'] = self.best_target_metric[i]['alpha']
                # val_update[f'model_{i}_nse'] = self.best_target_metric[i]['nse']
                
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
        outputs, targets = [], []
        if self.valid_data:
            for i in range(len(self.valid_data)):
                # outputs, targets = [], []
                trunk, branch, target = self.valid_data[i]
                trunk, branch, target = trunk.to(self.device), branch.to(self.device), target.to(self.device)
                # import pdb; pdb.set_trace()
                x = torch.concat([trunk, branch], dim=1)

                output = self.model(x).squeeze()
                
                # 这行用来计算反归一化前每个y分量的loss，per_target_loss用于选样本
                # criterion用法：loss_per_sample, per_target_loss = self.criterion(output, target)
                # @JL: XB revised this part
                # _, per_target_loss = self.criterion(output, target)
                # print(self.target_detransform(output).shape)
                output = output.detach()
                per_target_loss = -1 * module_metric.r_squared(self.target_detransform(output), self.target_detransform(target))

                # loss = loss_per_sample.mean()
                if self.target_detransform:
                    output = self.target_detransform(output)
                    target = self.target_detransform(target)
                
                # 这行用来计算反归一化后每个样本的loss，loss_per_sample用于report val loss
                # loss_per_sample, _ = self.criterion(output, target, weight=torch.tensor([1, 1]))
                loss_per_sample, _ = self.criterion(output, target)
                loss = loss_per_sample.mean()
                # if epoch == 100:
                #     print(loss.item())

                valid_count = (~torch.isnan(target)).sum(axis=0).cpu()
                per_target_loss = per_target_loss.detach().cpu()
                per_target_loss = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), per_target_loss)
                # if i == 103:
                    # print(valid_count)

                # outputs.append(output.detach())
                # targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data) + i, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('per_target_loss', per_target_loss)
                for met in self.metric_ftns:
                    metric_value = met(output, target)
                    metric_value = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), metric_value)
                    self.valid_metrics.update(met.__name__, metric_value)
                    # change the 30 below to something else
                    # self.valid_metrics._data[met.__name__][-1] = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), self.valid_metrics._data[met.__name__][-1])

            # print(self.valid_metrics._data['nse'])
            # all = self.valid_metrics._data['nse']
            # all = torch.stack(all,axis=0).numpy()
            # import pdb;pdb.set_trace()

        else:
            with torch.no_grad():
                for batch_idx, (trunk, branch, target) in enumerate(self.valid_data_loader):
                    trunk, branch, target = trunk.to(self.device), branch.to(self.device), target.to(self.device)

                    output = self.model(trunk, branch)
                    # loss, _ = self.criterion(output, target, weight=torch.tensor([1, 1]))
                    loss, _ = self.criterion(output, target)
                    # loss = criterion(output, target)
                    # loss = loss.mean()

                    outputs.append(output.detach())
                    targets.append(target.detach())

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    # self.valid_metrics.update('loss', loss.item())
                    
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
            if self.target_detransform:
                outputs = self.target_detransform(outputs)
                targets = self.target_detransform(targets)
            # loss, _ = self.criterion(outputs, targets, weight=torch.tensor([1, 1]))
            loss, _ = self.criterion(outputs, targets)
            self.valid_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
            # self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(raw=True)
    