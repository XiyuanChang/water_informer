import torch.nn.functional as F
import torch
import numpy as np

def nll_loss(output, target):
    return F.nll_loss(output, target)

# def mse_loss(output, target):
#     return F.mse_loss(output, target, reduction='mean')

def mse_loss(input, target, reduction='none', threshold=None, weight=None):
    # Ensure the target is a float tensor (to handle NaN)
    target = target.float()
    
    # Create a mask that will be True where target is not NaN
    mask = ~torch.isnan(target)
    
    # Apply the mask to both input and target to ignore NaNs in target
    filtered_input = torch.where(mask, input, torch.tensor(0.0, device=input.device))
    filtered_target = torch.where(mask, target, torch.tensor(0.0, device=target.device))
    
    # Compute the squared differences; masked out values will contribute 0 to the sum
    # diff = filtered_input - filtered_target
    # square_diff = diff ** 2
    diff = filtered_input - filtered_target
    # square_diff = F.cosine_embedding_loss(filtered_input, filtered_target, reduction='none')
    square_diff = diff ** 2
    
    
    if threshold:
        weighted_matrix = torch.where(target >= threshold[0], threshold[1], threshold[2])
        square_diff = torch.mul(weighted_matrix, square_diff)
    if weight is not None:
        weight = weight.to(square_diff.device)
        square_diff = square_diff * weight
    
    # Count the number of valid (non-NaN) elements per example in the batch
    valid_elements = mask.sum(dim=1, keepdim=True).float()
    
    # Avoid division by zero by setting no valid elements to 1
    valid_elements = torch.where(valid_elements > 0, valid_elements, torch.tensor(1.0, device=target.device))
    
    # Compute the MSE for each example in the batch and sum up the errors
    # import pdb; pdb.set_trace()
    loss_per_example = square_diff.sum(dim=1) / valid_elements.squeeze()
    # loss_per_example = square_diff.sum() / valid_elements.squeeze()
    
    valid_targets = mask.sum(dim=0).float()
    valid_targets = torch.where(valid_targets > 0, valid_targets, torch.tensor(1.0, device=target.device))
    loss_per_target = square_diff.sum(dim=0) / valid_targets
    
    if reduction == 'mean':
        return loss_per_example.mean()
    elif reduction == 'sum':
        return loss_per_example.sum()
    else:
        return loss_per_example, loss_per_target
    

def pbias_loss(input, target, reduction='none', threshold=None):
    # Ensure the target is a float tensor (to handle NaN)
    target = target.float()
    
    # Create a mask that will be True where target is not NaN
    mask = ~torch.isnan(target)
    
    # Apply the mask to both input and target to ignore NaNs in target
    filtered_input = torch.where(mask, input, torch.tensor(0.0, device=input.device))
    filtered_target = torch.where(mask, target, torch.tensor(0.0, device=target.device))
    
    # Compute the squared differences; masked out values will contribute 0 to the sum
    diff = torch.abs(filtered_input - filtered_target)
    square_diff = diff ** 2

    
    if threshold:
        weighted_matrix = torch.where(target >= threshold[0], threshold[1], threshold[2])
        square_diff = torch.mul(weighted_matrix, square_diff)
    
    # Count the number of valid (non-NaN) elements per example in the batch
    valid_elements = mask.sum(dim=1, keepdim=True).float()
    
    # Avoid division by zero by setting no valid elements to 1
    valid_elements = torch.where(valid_elements > 0, valid_elements, torch.tensor(1.0, device=target.device))
    
    # Compute the MSE for each example in the batch and sum up the errors
    # import pdb; pdb.set_trace()
    loss_per_example = square_diff.sum(dim=1) / valid_elements.squeeze()
    
    valid_targets = mask.sum(dim=0).float()
    valid_targets = torch.where(valid_targets > 0, valid_targets, torch.tensor(1.0, device=target.device))
    
    # loss_per_target = square_diff.sum(dim=0) / valid_targets
    loss_per_target = square_diff.sum(dim=0) / (torch.abs(filtered_target).sum(dim=0)+1e-8) 
    
    if reduction == 'mean':
        return loss_per_example.mean()
    elif reduction == 'sum':
        return loss_per_example.sum()
    else:
        return loss_per_example, loss_per_target