import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        # self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self._data = {key: None for key in keys}
        self.reset()

    def reset(self):
        for k in self._data:
            self._data[k] = []
        # for col in self._data.columns:
        #     self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
            # self.writer.add_scalar(key, value)
        self._data[key].append(value)
        # self._data.total[key] += value * n
        # self._data.counts[key] += n
        # self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        if key == 'loss':
            return np.mean(self._data[key])
        if not self._data[key]:
            return 0

        # return torch.nanmean(torch.stack(self._data[key], axis=0), dim=0)
        return torch.nanmedian(torch.stack(self._data[key], axis=0), dim=0).values
    
    def add_key(self, key):
        self._data[key] = []

    def result(self, raw=False):
        result = {}
        for k in self._data.keys():
            result[k] = self.avg(k)
        if raw:
            return result, self._data
        else:
            return result

# class MetricTracker:
#     def __init__(self, *keys, writer=None):
#         self.writer = writer
#         self._data = {}
#         for key in keys:
#             self._data[key] = {'last': None}
#         self.reset()

#     def reset(self):
#         for key in self._data:
#             self._data[key]['last'] = None

#     def update(self, key, value):
#         # Ensure the value is a NumPy array
#         if torch.is_tensor(value):
#             value = value.detach().cpu().numpy()  # Move tensor to CPU and convert to NumPy array

#         self._data[key]['last'] = value

#     def last(self, key):
#         return self._data[key]['last']

#     def result(self):
#         return {key: self._data[key]['last'] for key in self._data if self._data[key]['last'] is not None}

def remove_nan(x, remove_list: list):
    """ Removes the rows where x contains any NaN value in this row. The corresponding tensors' row in remove_list will be removed as well.
    Args:
        x (torch.Tensor): input x, of shape [N, Dx]
        remove_list: a list of tensors to be removed
    """
    mask = torch.isnan(x).any(dim=1)
    x = x[~mask]
    removed = [y[mask] for y in remove_list]
    return x, removed