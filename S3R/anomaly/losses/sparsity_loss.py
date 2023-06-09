import torch


def sparsity_loss(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss
