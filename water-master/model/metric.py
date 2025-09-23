import torch
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def _nse(predicted, observed):
    # Check if arrays are compatible
    if observed.shape != predicted.shape:
        raise ValueError("The shape of observed and predicted arrays must be the same.")

    # Create masks for valid (non-NaN) observations
    valid = ~torch.isnan(observed)
    if valid.sum() == 0:
        return torch.nan
    
    # Filter out invalid entries
    observed_valid = observed[valid]
    predicted_valid = predicted[valid]
    
    # Calculate the mean of the valid observed data
    mean_observed = torch.mean(observed_valid)
    
    # Calculate the residual sum of squares (numerator)
    numerator = torch.sum((observed_valid - predicted_valid) ** 2)+1e-5
    
    # Calculate the total sum of squares (denominator)
    denominator = torch.sum((observed_valid - mean_observed) ** 2)+1e-5
    
    # Compute NSE
    nse = 1 - (numerator / denominator)
    
    return nse.item()


def _kge(predicted, observed):
    # Check if arrays are compatible
    if observed.shape != predicted.shape:
        raise ValueError("The shape of observed and predicted arrays must be the same.")

    # Create masks for valid (non-NaN) observations
    valid = ~torch.isnan(observed)
    if valid.sum() == 0:
        return torch.nan
    if valid.sum() == 1:
        return torch.nan
    
    # Filter out invalid entries
    observed_valid = observed[valid]
    predicted_valid = predicted[valid]
    
    # Sim: predicitons
    # Obs: observations
    obs_mean = torch.mean(observed_valid)
    sim_mean = torch.mean(predicted_valid)
    obs_std = torch.std(observed_valid)
    sim_std = torch.std(predicted_valid)

    # Calculate correlation coefficient
    numerator = torch.sum((observed_valid - obs_mean) * (predicted_valid - sim_mean)) 
    denominator = torch.sqrt((torch.sum((observed_valid - obs_mean) ** 2) *
                   torch.sum((predicted_valid - sim_mean) ** 2)))
    
    # Compute R
    r = numerator / (denominator + 1e-6)
    
    # Calculate bias
    c = torch.mean(predicted_valid) / (torch.mean(observed_valid) + 1e-6)

    # Calculate variability ratio
    alpha = sim_std / (obs_std + 1e-6)

    # Calculate KGE
    kge = 1 - torch.sqrt((r - 1)**2 + (c - 1)**2 + (alpha - 1)**2)

    return kge.item()

def _beta(predicted, observed):
    # Check if arrays are compatible
    if observed.shape != predicted.shape:
        raise ValueError("The shape of observed and predicted arrays must be the same.")

    # Create masks for valid (non-NaN) observations
    valid = ~torch.isnan(observed)
    if valid.sum() == 0:
        return torch.nan
    if valid.sum() == 1:
        return torch.nan
    
    # Filter out invalid entries
    observed_valid = observed[valid]
    predicted_valid = predicted[valid]
    
    # Calculate bias
    c = torch.mean(predicted_valid) / (torch.mean(observed_valid) + 1e-6)

    return c.item()

def _alpha(predicted, observed):
    # Check if arrays are compatible
    if observed.shape != predicted.shape:
        raise ValueError("The shape of observed and predicted arrays must be the same.")

    # Create masks for valid (non-NaN) observations
    valid = ~torch.isnan(observed)
    if valid.sum() == 0:
        return torch.nan
    if valid.sum() == 1:
        return torch.nan
    
    # Filter out invalid entries
    observed_valid = observed[valid]
    predicted_valid = predicted[valid]
    
    # Sim: predicitons
    # Obs: observations
    obs_mean = torch.mean(observed_valid)
    sim_mean = torch.mean(predicted_valid)
    obs_std = torch.std(observed_valid)
    sim_std = torch.std(predicted_valid)

    
    # Calculate bias
    alpha = sim_std / (obs_std + 1e-6)

    return alpha.item()



def _r_squared(predicted, observed):
    # Check if arrays are compatible
    if observed.shape != predicted.shape:
        raise ValueError("The shape of observed and predicted arrays must be the same.")
    
    # Create masks for valid (non-NaN) observations
    valid = ~torch.isnan(observed)
    if valid.sum() == 0:
        return torch.nan
    
    # Filter out invalid entries
    observed_valid = observed[valid]
    predicted_valid = predicted[valid]
    
    # Calculate means of valid data
    mean_observed = torch.mean(observed_valid)
    mean_predicted = torch.mean(predicted_valid)
    
    # Calculate components of the R^2 formula
    numerator = torch.sum((observed_valid - mean_observed) * (predicted_valid - mean_predicted)) ** 2
    denominator = (torch.sum((observed_valid - mean_observed) ** 2) *
                   torch.sum((predicted_valid - mean_predicted) ** 2)) + 1e-6
    
    # Compute R^2
    r_squared = numerator / denominator
    
    return r_squared.item()


def _r(predicted, observed):
    # Check if arrays are compatible
    if observed.shape != predicted.shape:
        raise ValueError("The shape of observed and predicted arrays must be the same.")
    
    # Create masks for valid (non-NaN) observations
    valid = ~torch.isnan(observed)
    if valid.sum() == 0:
        return torch.nan
    
    # Filter out invalid entries
    observed_valid = observed[valid]
    predicted_valid = predicted[valid]
    
    # Calculate means of valid data
    mean_observed = torch.mean(observed_valid)
    mean_predicted = torch.mean(predicted_valid)
    
    # Calculate components of the r formula
    numerator = torch.sum((observed_valid - mean_observed) * (predicted_valid - mean_predicted)) 
    denominator = torch.sqrt(torch.sum((observed_valid - mean_observed) ** 2) *
                   torch.sum((predicted_valid - mean_predicted) ** 2)) + 1e-6
    
    # Compute R^2
    r = numerator / denominator
    
    return r.item()


def _pbias(predicted, observed):
    # Check if arrays are compatible
    if observed.shape != predicted.shape:
        raise ValueError("The shape of observed and predicted arrays must be the same.")
    
    # Create masks for valid (non-NaN) observations
    valid = ~torch.isnan(observed)
    if valid.sum() == 0:
        return torch.nan

    # Filter out invalid entries
    observed_valid = observed[valid]
    predicted_valid = predicted[valid]
    
    # Calculate the Percent Bias (PBIAS)
    numerator = torch.sum(observed_valid - predicted_valid)
    denominator = torch.sum(observed_valid) + 1e-6
    
    # Compute PBIAS
    pbias = torch.abs((numerator / denominator) * 100)
    
    return pbias.item()

# change _nse to _kge @JL

def kge(predictions, observations):
    
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    num_features = predictions.shape[1]
    nse_scores = torch.zeros(num_features)

    for i in range(num_features):
        nse_scores[i] = _kge(predictions[:, i], observations[:, i])

    return nse_scores
    # return torch.round(nse_scores*1000) / 1000


def beta(predictions, observations):
    
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    num_features = predictions.shape[1]
    nse_scores = torch.zeros(num_features)

    for i in range(num_features):
        nse_scores[i] = _beta(predictions[:, i], observations[:, i])

    return nse_scores

def alpha(predictions, observations):
    
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    num_features = predictions.shape[1]
    nse_scores = torch.zeros(num_features)

    for i in range(num_features):
        nse_scores[i] = _alpha(predictions[:, i], observations[:, i])

    return nse_scores



def nse(predictions, observations):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) for each feature.
    
    Parameters:
    predictions (torch.Tensor): Predicted values of shape [N, dim]
    observations (torch.Tensor): Observed values of shape [N, dim]
    
    Returns:
    torch.Tensor: NSE values for each feature of shape [dim]
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    num_features = predictions.shape[1]
    nse_scores = torch.zeros(num_features)

    for i in range(num_features):
        nse_scores[i] = _nse(predictions[:, i], observations[:, i])

    return nse_scores

def r_squared(predictions, observations):
    """
    Calculate the R-squared for each feature.
    
    Parameters:
    predictions (torch.Tensor): Predicted values of shape [N, dim]
    observations (torch.Tensor): Observed values of shape [N, dim]
    
    Returns:
    torch.Tensor: R-squared values for each feature of shape [dim]
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    num_features = predictions.shape[1]
    r_squared_scores = torch.zeros(num_features)
    
    for i in range(num_features):
        r_squared_scores[i] = _r_squared(predictions[:, i], observations[:, i])
        
    # return r_squared_scores
    return torch.round(r_squared_scores*1000) / 1000


def pbias(predictions, observations):
    """
    Calculate the Percent Bias (PBIAS) for each feature.
    
    Parameters:
    predictions (torch.Tensor): Predicted values of shape [N, dim]
    observations (torch.Tensor): Observed values of shape [N, dim]
    
    Returns:
    torch.Tensor: PBIAS values for each feature of shape [dim]
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    num_features = predictions.shape[1]
    pbias_scores = torch.zeros(num_features)
    
    for i in range(num_features):
        pbias_scores[i] = _pbias(predictions[:, i], observations[:, i])
        
    # return pbias_scores
    return torch.round(pbias_scores*1000) / 1000


def r(predictions, observations):
    """
    Calculate the R-squared for each feature.
    
    Parameters:
    predictions (torch.Tensor): Predicted values of shape [N, dim]
    observations (torch.Tensor): Observed values of shape [N, dim]
    
    Returns:
    torch.Tensor: R-squared values for each feature of shape [dim]
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    num_features = predictions.shape[1]
    r_squared_scores = torch.zeros(num_features)
    
    for i in range(num_features):
        r_squared_scores[i] = _r(predictions[:, i], observations[:, i])
        
    # return r_squared_scores
    return torch.round(r_squared_scores*1000) / 1000


# def kge(predictions, observations):
#     """
#     Calculate the Kling-Gupta Efficiency (KGE) for each feature.
    
#     Parameters:
#     predictions (torch.Tensor): Predicted values of shape [N, dim]
#     observations (torch.Tensor): Observed values of shape [N, dim]
    
#     Returns:
#     torch.Tensor: KGE values for each feature of shape [dim]
#     """
#     if isinstance(predictions, np.ndarray):
#         predictions = torch.from_numpy(predictions)
#     if isinstance(observations, np.ndarray):
#         observations = torch.from_numpy(observations)

#     num_features = predictions.shape[1]
#     kge_scores = torch.zeros(num_features)
    
#     for i in range(num_features):
#         kge_scores[i] = _kge(predictions[:, i], observations[:, i])
 
#     return kge_scores