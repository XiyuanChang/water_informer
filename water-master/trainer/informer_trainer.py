import numpy as np
import torch
from torchvision.utils import make_grid
from trainer.lstm_trainer import LSTMTrainer
from utils import inf_loop, MetricTracker, remove_nan
import torch.nn.functional as F
import json
import datetime
from collections import defaultdict
from tqdm import tqdm
import copy
import model.metric as module_metric

class InformerTrainer(LSTMTrainer):
    """
    Informer Trainer
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, valid_threshold=51, lr_scheduler=None, 
                 len_epoch=None, target_detransform=None, valid_data=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device,
                        data_loader, valid_data_loader, valid_threshold, lr_scheduler,
                        len_epoch, target_detransform, valid_data)
        
        # Informer specific parameters from config  
        self.seq_len = config.get('seq_len', 96)
        self.label_len = config.get('label_len', 48) 
        self.pred_len = config.get('pred_len', 1)  # Single-step prediction
        
        # Set default time features for Informer (will use dummy/zero time features)
        self.time_features = 3  # Standard time features for daily frequency
    
    def _prepare_informer_inputs(self, x, y):
        """
        Prepare encoder and decoder inputs for Informer
        Treat x as a whole, use dummy time features
        
        Args:
            x: input data [batch_size, time_len, features] 
            y: target data [batch_size, time_len, targets]
        Returns:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_target
        """
        batch_size, time_len = x.shape[0], x.shape[1]
     
        # Use x as a whole (no separation of time features)
        x_data = x  # [B, time_len, features]
        
        # Encoder inputs: use all timesteps EXCEPT the last one (we predict the last one)
        x_enc = x_data[:, :-1, :]  # [B, time_len-1, features] - exclude last timestep
        
        # Create dummy time features (zeros) for encoder
        x_mark_enc = torch.zeros(batch_size, time_len-1, self.time_features).to(x.device)
        
        # Target: predict the last timestep    
        y_target = y[:, -1:, :]  # [B, 1, targets] - only the last timestep
        
        # Decoder inputs: 
        # Label part: last label_len timesteps from encoder sequence
        start_idx = max(0, time_len - 1 - self.label_len)  # Ensure we don't go negative
        x_dec_label = x_data[:, start_idx:time_len-1, :]  # [B, label_len, features]
        
        # Prediction part: use ACTUAL input data for the timestep we're predicting
        x_dec_pred = x_data[:, -1:, :]  # [B, 1, features] - actual data for prediction timestep
        
        # Concatenate label and prediction parts
        x_dec = torch.cat([x_dec_label, x_dec_pred], dim=1)  # [B, label_len + 1, features]
        
        # Create dummy time features (zeros) for decoder
        x_mark_dec = torch.zeros(batch_size, x_dec.shape[1], self.time_features).to(x.device)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec, y_target

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch - adapted for Informer
        """
        print("Current time:", datetime.datetime.now().strftime("%H:%M:%S"))
        self.model.train()
        self.train_metrics.reset()
        outputs, targets = [], []

        for batch_idx, (x, y) in enumerate(self.data_loader):
            x, y = x.cuda(), y.cuda()
            self.optimizer.zero_grad()

            # Prepare Informer inputs
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_target = self._prepare_informer_inputs(x, y)
            
            # Informer specific forward pass
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, c_out]
            
            threshold = self.config['threshold']
            
            # Use the same loss function as LSTM - output is [B, 1, targets]
            _, loss_ = self.criterion(output, y_target, threshold=threshold)
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

            if epoch % self.log_epoch == 0:
                outputs.append(output[:, -1, :].detach().cpu())  # Last prediction timestep
                target = y_target[:, -1, :].cpu()  # Ground truth for last timestep
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
                    val_update['val_best_pbias'] = self._get_best_pbias(self.val_history[k])
                elif k == 'loss':
                    val_update['val_loss'] = v
                elif k == 'per_target_loss':
                    val_update['val_target_loss'] = v.tolist()
            
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

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch - adapted for Informer
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_metrics.add_key('per_target_loss')
        
        if self.config['x_feature']:
            x_feat = len(self.config['x_feature'])
        else:
            x_feat = 26

        if self.valid_data:
            # Station-based validation (similar to LSTM)
            for i in range(len(self.valid_data)):
                outputs, targets = [], []
                loss_per_sample_list, per_target_loss_list = [], []

                x, y = self.valid_data[i]
                x, y = x.to(self.device), y.to(self.device)
                
                for batch_idx in range(0, x.shape[0], 4096):
                    batch_x, batch_y = x[batch_idx:batch_idx+4096], y[batch_idx:batch_idx+4096]
                    
                    # Prepare Informer inputs
                    x_enc, x_mark_enc, x_dec, x_mark_dec, y_target = self._prepare_informer_inputs(batch_x, batch_y)
                    
                    pred = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    loss_per_sample, per_target_loss = self.criterion(pred, y_target)
                    
                    pred = pred.detach().cpu()
                    pred_y = pred[:, -1, :]  # Last prediction timestep
                    target = y_target[:, -1, :].cpu()  # Ground truth for last timestep
                    target = torch.where(target == -1, torch.tensor(float('nan')), target)
                    
                    outputs.append(pred_y)
                    targets.append(target)
                    loss_per_sample_list.append(loss_per_sample.detach().cpu())
                    per_target_loss_list.append(per_target_loss.detach().cpu())
                   
                loss_per_sample = torch.cat(loss_per_sample_list, axis=0)
                per_target_loss = torch.stack(per_target_loss_list, axis=0).sum(axis=0)
                
                target = torch.cat(targets, axis=0)
                pred_y = torch.cat(outputs, axis=0)

                loss = loss_per_sample.mean()

                if self.target_detransform:
                    pred_y = self.target_detransform(pred_y)
                    target = self.target_detransform(target)

                valid_count = (~torch.isnan(target)).sum(axis=0).cpu()
                per_target_loss = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), per_target_loss)

                self.writer.set_step((epoch - 1) * len(self.valid_data) + i, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('per_target_loss', per_target_loss)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_y, target), n=1)
                    self.valid_metrics._data[met.__name__][-1] = torch.where(valid_count < self.valid_threshold, torch.tensor(float('nan')), self.valid_metrics._data[met.__name__][-1])

        else:
            # DataLoader-based validation
            targets = defaultdict(list)
            predictions = defaultdict(list)
            masks = defaultdict(list)
            print("Validating...")

            with torch.no_grad():
                for batch_idx, (staids, x, y) in enumerate(tqdm(self.valid_data_loader)):
                    x, y = x.cuda(), y.cuda()

                    # Prepare Informer inputs
                    x_enc, x_mark_enc, x_dec, x_mark_dec, y_target = self._prepare_informer_inputs(x, y)
                    
                    output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    loss_per_sample, _ = self.criterion(output, y_target)
                    loss = torch.nanmean(loss_per_sample)
                    
                    pred = output.detach()
                    pred = pred[:, -1, :].cpu()  # Last prediction timestep
                    target = y_target[:, -1, :].cpu()  # Ground truth for last timestep
                    
                    # Check for valid samples using all input features
                    x_data = x[:, -1, :].cpu()  # Last timestep all features
                    mask = ~(x_data == -1).any(dim=1)
                    
                    for idx, staid in enumerate(staids):
                        targets[staid].append(target[idx])
                        predictions[staid].append(pred[idx])
                        masks[staid].append(mask[idx])
                    
                    self.valid_metrics.update('loss', loss.item())

            for staid in targets.keys():
                target = torch.stack(targets[staid], axis=0).cuda()
                pred = torch.stack(predictions[staid], axis=0).cuda()
                mask = torch.stack(masks[staid], axis=0).cuda()

                target = target[mask]
                pred = pred[mask]
                if len(target) == 0:
                    continue
                
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

    def _get_best_pbias(self, arr):
        """Helper function for best pbias calculation"""
        array = np.array(arr)
        abs_arr = np.abs(array)
        min_index = np.argmin(abs_arr, axis=0)
        min_abs_values = array[min_index, np.arange(array.shape[1])]
        return min_abs_values 