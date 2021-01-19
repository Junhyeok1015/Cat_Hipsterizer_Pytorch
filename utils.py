import torch


def save_checkpoint_bbs(epoch, model, optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'models/checkpoint_bbs.pth.tar'
    torch.save(state, filename)

def save_checkpoint_lmks(epoch, model, optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'models/checkpoint_lmks.pth.tar'
    torch.save(state, filename)