import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

# def mse_loss(output, target):
#     return F.mse_loss(output, target, reduction='mean')

def mse_loss(input, target, reduction='none', threshold=None):
    # Change the target dtype to float for operations
    target = target.float()
    
    # Create a mask that will be True where the target is not -1
    # mask = target != -1
    mask = ~torch.isnan(target)
    
    # Apply the mask to both input and target to ignore -1 values in target
    filtered_input = torch.where(mask, input, torch.tensor(0.0, device=input.device))
    filtered_target = torch.where(mask, target, torch.tensor(0.0, device=target.device))

    # Compute the squared differences; -1 values will contribute 0 to the calculations
    diff = filtered_input - filtered_target
    square_diff = diff ** 2
    # square_diff = torch.abs(diff)

    # Apply weights based on the threshold
    if threshold:
        weight_matrix = torch.where(target >= threshold[0], threshold[1], threshold[2])
        square_diff = torch.mul(weight_matrix, square_diff)
    
    # Count the number of valid (non--1) elements in each feature
    valid_elements_by_feature = mask.sum(dim=(0, 1)).float()  # Sum over batch and sequence dimensions
    
    # Avoid division by zero by setting no valid elements to 1
    valid_elements_by_feature = torch.where(valid_elements_by_feature > 0, valid_elements_by_feature, torch.tensor(1.0, device=target.device))
    
    # Compute the MSE for each feature across all batches and sequences
    loss_per_feature = square_diff.sum(dim=(0, 1)) / valid_elements_by_feature
    
    if reduction == 'mean':
        # Compute the total mean loss by summing over the features and dividing by the number of features
        loss = loss_per_feature.mean()
    elif reduction == 'sum':
        # Compute the total loss by summing over the features
        loss = loss_per_feature.sum()
    else:
        # Return loss per batch example and per feature
        valid_elements = mask.sum(dim=(1, 2)).float()  # Count per batch example
        valid_elements = torch.where(valid_elements > 0, valid_elements, torch.tensor(1.0, device=target.device))
        loss_per_example = square_diff.sum(dim=(1, 2)) / valid_elements
        return loss_per_example, loss_per_feature
    return loss