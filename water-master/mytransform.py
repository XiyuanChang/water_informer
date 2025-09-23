import torchvision.transforms as transforms
import torch
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import QuantileTransformer

BRANCH_MAX = [206.7560,  41.6667,  30.5556,   1.0000,  14.3008,  30.2486]
BRANCH_MIN = [  0.0000, -26.6667, -32.2222,   0.1915,   0.0000,   0.0000]
BRANCH_MEAN = [ 2.6407, 15.5155,  4.7881,  0.6833,  3.9973, 14.7349]
BRANCH_STD = [ 7.0767, 11.5181, 10.0697,  0.1015,  1.6976,  7.4404]
BRANCH_MAXV2 = [1.9900e+05, 5.9933e+01, 2.2182e+02, 2.1789e-02, 3.9984e+02, 3.0094e+02,
        3.1951e+02, 1.4887e+01, 2.3086e+01, 8.5300e+00, 1.3908e+03, 1.2568e+02,
        1.2896e+01, 2.7857e+01, 5.3004e+01, 4.0880e+01, 5.6880e+01, 8.0570e+01,
        3.6376e+02, 2.9999e+05, 6.1771e+02, 5.8166e+02, 5.6931e+03, 6.9390e+03,
        1.0000e+00, 1.0000e+00]
BRANCH_MINV2 = [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  2.4399e-04,  0.0000e+00,
         2.3480e+02,  2.5080e+02,  0.0000e+00,  0.0000e+00,  2.9000e+00,
         1.2000e+00,  2.0000e-03,  1.0000e-03,  1.0000e-03,  1.0000e-03,
         4.0000e-03,  5.0000e-03,  3.0000e-03,  4.0000e-03,  1.4701e+02,
        -1.5486e+00, -1.0085e+00, -2.8980e+01, -6.5740e+03, -1.0000e+00,
        -1.0000e+00]
BRANCH_STDV2 = [5.1907e+03, 1.4322e+00, 9.7799e+00, 4.0071e-03, 8.5183e+01, 9.2188e+00,
        1.0122e+01, 1.8173e+00, 2.3543e+00, 7.5985e-01, 1.9275e+01, 8.6001e-01,
        1.5323e-01, 1.4390e-01, 5.5463e-01, 5.5292e-01, 1.6750e+00, 8.1849e-01,
        2.1299e+00, 6.0442e+04, 1.3225e+02, 5.6030e+01, 5.8695e+02, 3.7112e+03,
        7.1895e-01, 6.8773e-01]
BRANCH_MEANV2 = [ 1.6331e+03,  5.9620e-01,  3.9746e+00,  6.4174e-03,  1.8942e+02,
         2.7789e+02,  2.9004e+02,  3.2020e+00,  4.2819e+00,  5.1707e+00,
         1.7042e+01,  3.3930e-01,  5.3184e-02,  4.5015e-02,  1.6796e-01,
         4.2497e-01,  1.5292e+00,  2.7161e-01,  1.5691e+00,  9.4287e+04,
         1.6548e+02,  1.0041e+02,  2.7541e+02,  7.9777e+02,  5.4801e-02,
        -8.4504e-02]

TRUNC_MAX = [ 4.1993e+01, -8.3453e+01,  3.2467e+04,  4.5607e+00,  3.3059e+04,
         3.0937e+00,  4.1444e+01,  1.3138e+00,  3.3534e+02,  6.0435e+01,
         5.3939e+01,  9.6480e+01,  1.0980e+01,  9.4740e+01,  3.3880e+01,
         4.9870e+01,  3.6420e+01,  4.4540e+01,  6.5800e+00,  4.8310e+01,
         7.7300e+00,  2.4200e+00,  2.3600e+00,  2.5500e+00,  2.9190e+01,
         2.7970e+01,  1.0660e+01,  3.2900e+00,  2.8900e+00,  1.1140e+01,
         7.2200e+00,  2.6000e+00,  8.6000e-01,  1.8100e+00,  1.5210e+01,
         6.5900e+00,  1.2080e+01,  5.2200e+00,  4.0900e+00,  2.3700e+00,
         3.9000e-01,  3.7700e+00,  1.7500e+00,  2.5400e+00,  2.7900e+00,
         9.3500e+00,  5.7600e+00,  5.4700e+00,  2.2940e+01,  1.1940e+01,
         8.1900e+00,  1.4300e+00,  3.5500e+00,  2.5070e+01,  8.3500e+00,
         1.7610e+01,  7.0400e+00,  8.2900e+00,  5.1800e+00,  7.1000e-01,
         1.0270e+01,  2.3900e+00,  5.6300e+00,  1.8800e+00]
TRUNC_MIN = [ 4.0457e+01, -8.5249e+01,  1.3650e+00,  2.7710e-01,  1.8953e+02,
         2.1203e-02,  9.8101e-02,  2.3334e-02,  1.7589e+02,  1.2515e-01,
         3.8911e-01,  4.1822e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]
TRUNC_MEAN = [ 4.1184e+01, -8.4380e+01,  6.0001e+03,  1.5349e+00,  1.4901e+04,
         5.2650e-01,  1.3867e+01,  6.1151e-01,  2.4227e+02,  3.0030e+01,
         4.1132e+01,  2.6253e+01,  4.3083e+00,  2.4536e+00,  4.7191e+00,
         4.9697e+00,  3.3856e+00,  1.8034e+00,  1.6379e-01,  6.7616e+00,
         1.0453e-01,  1.2985e-01,  5.5764e-02,  3.2995e-01,  3.9207e+00,
         3.1836e+00,  4.7325e-01,  5.1345e-01,  4.9074e-01,  3.4442e+00,
         1.7317e+00,  6.2320e-01,  6.9360e-02,  2.5764e-01,  4.9131e+00,
         1.4999e+00,  2.8160e+00,  5.5468e-01,  5.0438e-01,  4.5498e-01,
         3.3941e-02,  6.2601e-01,  1.9724e-01,  4.9433e-01,  1.1591e-01,
         3.0422e+00,  6.7956e-01,  8.3635e-01,  8.1073e+00,  3.7031e+00,
         1.4280e+00,  1.1754e-01,  3.9645e-01,  1.0764e+01,  3.1397e+00,
         4.8926e+00,  1.2102e+00,  1.3942e+00,  9.4020e-01,  5.7438e-02,
         1.6757e+00,  4.0394e-01,  8.9502e-01,  2.0877e-01]
TRUNC_STD = [3.5261e-01, 4.4293e-01, 4.8902e+03, 8.8419e-01, 7.4598e+03, 4.9303e-01,
        7.0920e+00, 2.2585e-01, 3.1338e+01, 1.1038e+01, 1.0563e+01, 1.4404e+01,
        2.1652e+00, 8.7676e+00, 3.8943e+00, 6.1119e+00, 6.4171e+00, 5.3086e+00,
        6.2856e-01, 5.2631e+00, 5.7562e-01, 2.6409e-01, 2.3809e-01, 3.6242e-01,
        5.5361e+00, 4.7814e+00, 9.9249e-01, 5.5093e-01, 4.4564e-01, 2.3899e+00,
        1.2168e+00, 5.1391e-01, 9.7010e-02, 2.6619e-01, 3.0794e+00, 1.2029e+00,
        2.3075e+00, 6.7950e-01, 5.6554e-01, 4.7142e-01, 5.0760e-02, 6.1458e-01,
        2.4295e-01, 5.6800e-01, 2.3641e-01, 1.7745e+00, 9.2544e-01, 9.2130e-01,
        5.6121e+00, 2.3509e+00, 1.4016e+00, 1.7402e-01, 5.7635e-01, 6.1672e+00,
        2.0169e+00, 3.3905e+00, 1.4036e+00, 1.7140e+00, 9.4382e-01, 1.1250e-01,
        1.8589e+00, 4.1953e-01, 9.6160e-01, 3.0627e-01]

# with open("trunk_statv2.pt", "rb") as f:
#     stats = torch.load(f)
#     TRUNC_MAXV2 = stats["max"]
#     TRUNC_MINV2 = stats["min"]
#     TRUNC_MEANV2 = stats["mean"]
#     TRUNC_STDV2 = stats["std"]
# with open("target_statv2.pt", "rb") as f:
#     stats = torch.load(f)
#     TARGET_MAXV2 = stats["max"]
#     TARGET_MINV2 = stats["min"]
#     TARGET_MEANV2 = stats["mean"]
#     TARGET_STDV2 = stats["std"]

TARGET_MAX = [4450]
TARGET_MIN = [1.6800e-09]
TARGET_MEAN = [2.0341]
TARGET_STD = [21.1609]

class MinMaxNormalize(object):
    """
    data - min(data) / max(data) - min(data)
    """
    def __init__(self, max, min):
        self.max = max
        self.min = min
        if not isinstance(self.max, torch.Tensor):
            self.max = torch.tensor(max, dtype=torch.float32)
        if not isinstance(self.min, torch.Tensor):
            self.min = torch.tensor(min, dtype=torch.float32)

    def __call__(self, tensor):
        # Perform the normalization using PyTorch operations
        if self.max.device != tensor.device:
            self.max = self.max.to(tensor.device)
            self.min = self.min.to(tensor.device)
        normalized_tensor = (tensor - self.min) / (self.max - self.min)

        return normalized_tensor


class MinMaxNormalizeReal(object):
    """
    data - min(data) / max(data) - min(data)
    """
    def __init__(self, min, max):
        self.max = max
        self.min = min
        if not isinstance(self.max, torch.Tensor):
            self.max = torch.tensor(max, dtype=torch.float32)
        if not isinstance(self.min, torch.Tensor):
            self.min = torch.tensor(min, dtype=torch.float32)

    def __call__(self, tensor):
        # Perform the normalization using PyTorch operations
        if self.max.device != tensor.device:
            self.max = self.max.to(tensor.device)
            self.min = self.min.to(tensor.device)
        normalized_tensor = (tensor - self.min) / (self.max - self.min)

        return normalized_tensor

class MeanStdNormalize(object):
    """
    data - mean(data) / std(data)
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, tensor):
        if self.mean.device != tensor.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)
        # Perform the normalization using PyTorch operations
        normalized_tensor = (tensor - self.mean) / self.std

        return normalized_tensor

class MinMaxDeNormalize(object):
    """
    data - min(data) / max(data) - min(data)
    """
    def __init__(self, min, max):
        self.max = max
        self.min = min
        if not isinstance(self.max, torch.Tensor):
            self.max = torch.tensor(max, dtype=torch.float32)
        if not isinstance(self.min, torch.Tensor):
            self.min = torch.tensor(min, dtype=torch.float32)

    def __call__(self, tensor):
        # Perform the normalization using PyTorch operations
        if self.max.device != tensor.device:
            self.max = self.max.to(tensor.device)
            self.min = self.min.to(tensor.device)
        normalized_tensor = tensor * (self.max - self.min) + self.min

        return normalized_tensor

class MeanStdDeNormalize(object):
    """
    data - mean(data) / std(data)
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, tensor):
        # Perform the normalization using PyTorch operations
        if self.mean.device != tensor.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)
        normalized_tensor = tensor * self.std + self.mean

        return normalized_tensor

class LogNormalize(object):
    def __init__(self, stats, index=None):
        self.stats = stats
        self.index = index
        self.tenth = stats['10th'].squeeze()
        self.ninetieth = stats['90th'].squeeze()
        self.log_min = stats['logmin'].squeeze()
        self.log_max = stats['logmax'].squeeze()
        self.min = stats['min'].squeeze()
        self.max = stats['max'].squeeze()

    def __call__(self, tensor):
        if self.tenth.device != tensor.device:
            self.tenth = self.tenth.to(tensor.device)
            self.ninetieth = self.ninetieth.to(tensor.device)
            self.log_min = self.log_min.to(tensor.device)
            self.log_max = self.log_max.to(tensor.device)
            self.min = self.min.to(tensor.device)
            self.max = self.max.to(tensor.device)

        log_tensor = torch.log(tensor.clamp(min=1e-9))
        # print(f"tensor size: {tensor.shape}\n")

        log_normalized_tensor = (log_tensor - self.log_min) / (self.log_max - self.log_min)
        if self.index is None or len(self.index) == 0:
            return log_normalized_tensor

        minmax_normalized = (tensor - self.min) / (self.max - self.min)
        if tensor.dim() == 1:
            for idx in self.index:
                log_normalized_tensor[idx] = minmax_normalized[idx]
        elif tensor.dim() == 2:
            for idx in self.index:
                log_normalized_tensor[:, idx] = minmax_normalized[:, idx]
        # numerator = log_tensor - self.tenth
        # denominator = self.ninetieth - self.tenth
        # denominator[denominator == 0] = 1e-9
        # normalized_tensor = (numerator / denominator) * 2 - 1
        # make sure that the place where tensor is nan is identical to the place where log_normalized_tensor is nan
        assert torch.isnan(tensor).eq(torch.isnan(log_normalized_tensor)).all()

        return log_normalized_tensor

class LogDeNormalize(object):
    def __init__(self, stats, index=None):
        self.stats = stats
        self.index = index
        self.log_min = stats['logmin'].squeeze()
        self.log_max = stats['logmax'].squeeze()
        self.min = stats['min'].squeeze()
        self.max = stats['max'].squeeze()
    
    def __call__(self, y):
        if self.log_min.device != y.device:
            self.log_min = self.log_min.to(y.device)
            self.log_max = self.log_max.to(y.device)
            self.min = self.min.to(y.device)
            self.max = self.max.to(y.device)

        # Reverse log normalization
        y_star = y * (self.log_max - self.log_min) + self.log_min
        y_original = torch.exp(y_star)

        if self.index is None or len(self.index) == 0:
            return y_original

        minmax_denormalized = y * (self.max - self.min) + self.min

        if y.dim() == 1:
            for idx in self.index:
                y_original[idx] = minmax_denormalized[idx]
        elif y.dim() == 2:
            for idx in self.index:
                y_original[:, idx] = minmax_denormalized[:, idx]

        return y_original
    
class LogMinMax(object):
    '''Perform log then minmax to the first three columns of the data, the rest columns are minmax normalized'''
    def __init__(self, stat, index=None):
        self.log_max = stat['logmax']
        self.log_min = stat['logmin']
        self.min = stat['min']
        self.max = stat['max']
        self.index = index
        # print(self.index)
        self.quantile_init_params = stat['quantile_init_params']
        self.quantile_fit_params = stat['quantile_fit_params']
        self.quantile_init_params_log = stat['quantile_init_params_log']
        self.quantile_fit_params_log = stat['quantile_fit_params_log']

        if not isinstance(self.log_max, torch.Tensor):
            self.log_max = torch.tensor(self.log_max, dtype=torch.float32)
        if not isinstance(self.log_min, torch.Tensor):
            self.log_min = torch.tensor(self.log_min, dtype=torch.float32)
        if not isinstance(self.min, torch.Tensor):
            self.min = torch.tensor(self.min, dtype=torch.float32)
        if not isinstance(self.max, torch.Tensor):
            self.max = torch.tensor(self.max, dtype=torch.float32)
    
    def __call__(self, x):
        if self.log_max.device != x.device:
            self.log_max = self.log_max.to(x.device)
            self.log_min = self.log_min.to(x.device)
            self.min = self.min.to(x.device)
            self.max = self.max.to(x.device)
        
        log_x = torch.log(x.clamp(min=1e-9))
        if x.device != torch.device('cpu'):
            x = x.cpu()
            log_x = log_x.cpu()
        if x.dim() == 1:
            new_x = x.unsqueeze(0)
            log_x = log_x.unsqueeze(0)
        else:
            new_x = x

        # minmax normalize log_x
        log_normalized = (log_x - self.log_min) / (self.log_max - self.log_min)
        
        # scaler = CustomQuantileTransformer(output_distribution='uniform')
        # scaler.set_init_params(self.quantile_init_params_log)
        # scaler.set_fitted_params(self.quantile_fit_params_log)
        # log_normalized = scaler.transform(log_x.numpy())
        if x.dim() == 1:
            log_normalized = log_normalized.squeeze()
        # log_normalized = torch.tensor(log_normalized, dtype=torch.float32)
        if log_normalized.device != self.log_max.device:
            log_normalized = log_normalized.to(self.log_max.device)

        if self.index is None or len(self.index) == 0:
            return log_normalized
        
        minmax_normalized = (new_x - self.min) / (self.max - self.min)

        # scaler = CustomQuantileTransformer(output_distribution='uniform')
        # scaler.set_init_params(self.quantile_init_params)
        # scaler.set_fitted_params(self.quantile_fit_params)
        
        # minmax_normalized = scaler.transform(new_x.numpy())
        if x.dim() == 1:
            minmax_normalized = minmax_normalized.squeeze()
        # minmax_normalized = torch.tensor(minmax_normalized, dtype=torch.float32)
        if minmax_normalized.device != self.log_max.device:
            minmax_normalized = minmax_normalized.to(self.log_max.device)

        assert log_normalized.dim() == minmax_normalized.dim()
        if log_normalized.dim() == 1:
            for idx in self.index:
                log_normalized[idx] = minmax_normalized[idx]
        elif log_normalized.dim() == 2:
            for idx in self.index:
                log_normalized[:, idx] = minmax_normalized[:, idx]
        else:
            raise ValueError("Unsupported tensor shape")

        return log_normalized

class InverseLogMinMax(object):
    '''Inverse of LogMinMax transform'''
    def __init__(self, stat, index=None):
        self.log_max = stat['logmax']
        self.log_min = stat['logmin']
        self.min = stat['min']
        self.max = stat['max']
        self.index = index
        
        if not isinstance(self.log_max, torch.Tensor):
            self.log_max = torch.tensor(self.log_max, dtype=torch.float32)
        if not isinstance(self.log_min, torch.Tensor):
            self.log_min = torch.tensor(self.log_min, dtype=torch.float32)
        if not isinstance(self.min, torch.Tensor):
            self.min = torch.tensor(self.min, dtype=torch.float32)
        if not isinstance(self.max, torch.Tensor):
            self.max = torch.tensor(self.max, dtype=torch.float32)
    
    def __call__(self, y):
        # Ensure parameters are on the same device as y
        if self.log_max.device != y.device:
            self.log_max = self.log_max.to(y.device)
            self.log_min = self.log_min.to(y.device)
            self.min = self.min.to(y.device)
            self.max = self.max.to(y.device)
        
        # Ensure y is at least 2D
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        # Get indices for columns
        total_features = y.shape[1]
        if self.index is None or len(self.index) == 0:
            indices_in_index = []
            indices_not_in_index = list(range(total_features))
        else:
            indices_in_index = self.index
            indices_not_in_index = [i for i in range(total_features) if i not in self.index]

        x = torch.zeros_like(y)

        # Inverse transform for indices not in self.index
        if indices_not_in_index:
            y_not_in_index = y[:, indices_not_in_index]
            log_min = self.log_min[indices_not_in_index]
            log_max = self.log_max[indices_not_in_index]
            # Inverse min-max normalization
            log_x = y_not_in_index * (log_max - log_min) + log_min
            # Exponentiate to get x
            x_not_in_index = torch.exp(log_x)
            x[:, indices_not_in_index] = x_not_in_index

        # Inverse transform for indices in self.index
        if indices_in_index:
            y_in_index = y[:, indices_in_index]
            min_val = self.min[indices_in_index]
            max_val = self.max[indices_in_index]
            # Inverse min-max normalization
            x_in_index = y_in_index * (max_val - min_val) + min_val
            x[:, indices_in_index] = x_in_index

        if squeezed:
            x = x.squeeze(0)

        return x

class CustomQuantileTransformer:
    def __init__(self, output_distribution='uniform', n_quantiles=1000, subsample=100000, random_state=None):
        # Store the initialization parameters
        self.transformer = QuantileTransformer(
            output_distribution=output_distribution, 
            n_quantiles=n_quantiles, 
            subsample=subsample, 
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, X):
        """Fit the quantile transformer on X"""
        self.transformer.fit(X)
        self.is_fitted = True

    def set_fitted_params(self, fitted_params):
        """
        Sets the internal fitted parameters like `n_quantiles_`, `quantiles_`, etc.
        """
        if fitted_params:
            self.transformer.n_quantiles_ = fitted_params['n_quantiles_']
            self.transformer.quantiles_ = fitted_params['quantiles_']
            self.transformer.references_ = fitted_params['references_']
            # self.transformer.fitted_ = fitted_params['fitted_']
            self.is_fitted = True

    def get_fitted_params(self):
        """
        Get the fitted parameters of a trained QuantileTransformer.
        These include internal states like `quantiles_`, `n_quantiles_`, etc.
        """
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet.")
        return {
            'n_quantiles_': self.transformer.n_quantiles_,
            'quantiles_': self.transformer.quantiles_,
            'references_': self.transformer.references_,
            # 'fitted_': self.transformer.fitted_
        }

    def set_init_params(self, init_params):
        """
        Set the initialization parameters for the QuantileTransformer.
        This includes user-provided parameters like `n_quantiles`, `subsample`, etc.
        """
        self.transformer.set_params(**init_params)

    def get_init_params(self):
        """
        Get the initialization parameters of the transformer (e.g., n_quantiles, output_distribution).
        These are user-provided at initialization.
        """
        return self.transformer.get_params()

    def transform(self, X):
        """Transform new data using the fitted quantile transformer"""
        if not self.is_fitted:
            raise ValueError("QuantileTransformer is not yet fitted.")
        return self.transformer.transform(X)

    def inverse_transform(self, X):
        """Inverse transform data"""
        if not self.is_fitted:
            raise ValueError("QuantileTransformer is not yet fitted.")
        return self.transformer.inverse_transform(X)
